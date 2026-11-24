# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""OPN (序盤専用の副 net) の静的評価を、 篩の局面 (m2_games.jsonl の約 13 万局面) に python で付ける。 2026-11-13
   G1 の測り方 (b): エンジンを建てずに、 学習器が書いた quantised.bin (rshogi 形式) をエンジンと同じ整数演算で順伝播する。

■ 何をするか
   1. quantised.bin を読む (FT は LEB128 / i16、 LayerStack 8 本)。 PSQT / threat / 追加 section がある net は止める。
   2. sfen -> HalfKA_hm の添字 (tools/training/bullet_inference_v19.py の encode_halfka = 学習器 shogi_halfka.rs の移植)。
   3. 整数演算: acc = bias + Σ行 (i16) -> clamp [0,254] -> 要素積 >>9 (u8) -> fc0 (i8 / i32) ->
      [sqr: min(127, v*v >> 19), lin: clamp(v >> 6, 0, 127)] -> fc1 -> clamp(>>6) -> fc2 + skip (fc0 の最後の行)。
      p2nnue.cpp の propagate と同じ式。 bucket は m2 の "bucket" (本番エンジンが付けた progress8kpabs の値) を使う。
   4. cp = raw / eff_fv (C の 0 方向の整数除算)。 eff_fv も m2 の値 (本番の FV_SCALE 26/28 の動的設定) を使う
      = 「この net を本番の exe と設定で eval したら出る値」と同じ目盛り (篩の |cp| > 1500 除外が同じ意味になる)。
   5. 篩のキャッシュ E:/shogi/p2shogi/tmp_mob/v4evalana/screen_<label>.jsonl ({"s": cp} を m2 と同じ順) を書く。
      この後 `py -3 screen_future.py <label> <net>` を呼べば、 キャッシュがあるのでエンジンは起動されない。

■ 検算 (--nagisa): NAGISA v3.1 の nn.bin (幅 1024 / threat なし / PSQT なし / QA 127 保存 -> x2) を同じ器で順伝播し、
   m5_v31.jsonl の s31 (NAGISA の exe が eval で出した値、 FV 28 固定) と局面ごとに比べる。 完全一致なら符号化と順伝播が正しい。

usage:
   py -3 opn_static.py net   <label> <quantised.bin>          # キャッシュを書く (既にあれば止める)
   py -3 opn_static.py check <quantised.bin> [N=2000]          # 先頭 N 局面だけ計算して分布を出す (キャッシュは書かない)
   py -3 opn_static.py nagisa [N=0]                            # v3.1 で検算 (N=0 は全局面)
"""
import ctypes, io, json, os, re, sys, time
import numpy as np

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE (対局の CPU を乱さない)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from bullet_inference_v19 import encode_halfka  # noqa: E402  (学習器 shogi_halfka.rs の移植)

M2 = "E:/shogi/p2shogi/tmp_mob/v4evalana/m2_games.jsonl"
M5 = "E:/shogi/p2shogi/tmp_mob/v4evalana/m5_v31.jsonl"
CACHE = "E:/shogi/p2shogi/tmp_mob/v4evalana/screen_%s.jsonl"
NAGISA = "E:/shogi/engines/nagisa_v31/NAGISA_V3/eval/nn.bin"
IN_DIM, NB = 73305, 8
MAGIC = b"COMPRESSED_LEB128"
pad32 = lambda n: (n + 31) // 32 * 32


# ---------------------------------------------------------------- net の読み込み
def leb_decode(buf):
    last = (buf & 0x80) == 0
    ends = np.flatnonzero(last)
    starts = np.empty_like(ends); starts[0] = 0; starts[1:] = ends[:-1] + 1
    ln = ends - starts + 1
    out = np.zeros(len(ends), dtype=np.int64)
    for k in range(1, int(ln.max()) + 1):
        sel = ln >= k
        out[sel] |= (buf[starts[sel] + k - 1].astype(np.int64) & 0x7F) << (7 * (k - 1))
    shift = 7 * ln
    out -= ((out >> (shift - 1)) & 1) * (np.int64(1) << shift)
    return out


def read_block(b, p):
    assert bytes(b[p:p + 17]) == MAGIC, "LEB128 magic が無い @ %d" % p
    n = int(b[p + 17:p + 21].view(np.uint32)[0])
    return leb_decode(b[p + 21:p + 21 + n]), p + 21 + n


def load_net(path, nagisa=False):
    """返り値: dict(W, L1, L2, ftb[W] i32, ftw[IN, W] i16, stacks[b] = (B0, W0, B1, W1, B2, W2))"""
    b = np.fromfile(path, dtype=np.uint8)
    alen = int(b[8:12].view(np.uint32)[0])
    arch = bytes(b[12:12 + alen]).decode("utf-8", "replace")
    p = 12 + alen + 4                                     # + ft_hash
    if nagisa:
        W, L1, L2, nstk_file = 1024, 16, 64, 9
    else:
        for bad in ("PSQT=", "Threat", "HandCountDense", "PIN=", "DUAL=", "AUX", "MOB=", "LSMODE=", "L4=", "EGTMP"):
            if bad in arch:
                raise SystemExit("🚨この器は PSQT / threat / 追加 section の無い net 専用: arch に %s がある\n%s" % (bad, arch))
        m = re.search(r"Features=HalfKA_hm\(Friend\)\[(\d+)->(\d+)x2\]", arch)
        W = int(m.group(2))
        L2 = int(re.search(r"AffineTransform\[1<-(\d+)\]", arch).group(1))
        L1 = int(re.search(r"AffineTransform\[(\d+)<-%d\]" % (2 * W), arch).group(1))
        nstk_file = NB
    ftb, p = read_block(b, p)
    ftw, p = read_block(b, p)
    assert len(ftb) == W and len(ftw) == IN_DIM * W, "FT の要素数 %d / %d" % (len(ftb), len(ftw))
    if nagisa:                                            # NAGISA は QA=127 で保存し読み込み時に x2 (convert_nagisa_net.py)
        ftb, ftw = ftb * 2, ftw * 2
    assert np.abs(ftw).max() <= 32767
    M = L1 - 1; L2IN = 2 * M
    ls = 4 + 4 * L1 + L1 * pad32(W) + 4 * L2 + L2 * pad32(L2IN) + 4 + pad32(L2)
    rest = len(b) - p
    assert rest == ls * nstk_file, "LayerStack の残り %d が %d x %d と合わない (arch: %s)" % (rest, ls, nstk_file, arch)
    st = []
    for k in range(NB):
        q = p + k * ls + 4
        rd32 = lambda n: (b[q:q + 4 * n].view(np.int32).astype(np.int64), q + 4 * n)
        rd8 = lambda r, c: (b[q:q + r * c].view(np.int8).reshape(r, c).astype(np.int64), q + r * c)
        B0, q = rd32(L1); W0, q = rd8(L1, pad32(W)); B1, q = rd32(L2); W1, q = rd8(L2, pad32(L2IN))
        B2, q = rd32(1); W2, q = rd8(1, pad32(L2))
        st.append((B0, W0[:, :W], B1, W1[:, :L2IN], int(B2[0]), W2[0, :L2]))
    return dict(W=W, L1=L1, L2=L2, ftb=ftb.astype(np.int32), ftw=ftw.astype(np.int16).reshape(IN_DIM, W), st=st, arch=arch)


# ---------------------------------------------------------------- sfen -> 盤面 (bullet_inference_v19 の state と同じ形)
PT = {"P": 1, "L": 2, "N": 3, "S": 4, "B": 5, "R": 6, "G": 7, "K": 8}
PROMO = {1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14}


def sfen_state(sfen):
    """YO の升番号: sq = (筋-1)*9 + (段-1)。 sfen の盤は 1 段目から、 各段は 9 筋 -> 1 筋の順。"""
    bd, stm, hand = sfen.split()[:3]
    board = [(0, 0)] * 81; bk = wk = 81
    for r, row in enumerate(bd.split("/")):
        f = 8; prom = False
        for ch in row:
            if ch.isdigit():
                f -= int(ch); continue
            if ch == "+":
                prom = True; continue
            c = 0 if ch.isupper() else 1; pt = PT[ch.upper()]
            if prom:
                pt = PROMO[pt]; prom = False
            sq = f * 9 + r
            board[sq] = (c, pt)
            if pt == 8:
                if c == 0: bk = sq
                else: wk = sq
            f -= 1
        assert f == -1, "sfen の段 %d が 9 升でない: %s" % (r + 1, sfen)
    hb = {k: 0 for k in range(1, 8)}; hw = {k: 0 for k in range(1, 8)}
    if hand != "-":
        for n, ch in re.findall(r"(\d*)([A-Za-z])", hand):
            (hb if ch.isupper() else hw)[PT[ch.upper()]] += int(n) if n else 1
    return dict(board=board, black_hand=hb, white_hand=hw, side_to_move=0 if stm == "b" else 1,
                black_king_sq=bk, white_king_sq=wk)


# ---------------------------------------------------------------- 順伝播 (p2nnue.cpp と同じ整数演算)
def forward(net, sfens, buckets, chunk=1000):
    W, M = net["W"], net["L1"] - 1
    ftw, ftb = net["ftw"], net["ftb"]
    raw = np.zeros(len(sfens), dtype=np.int64)
    for c0 in range(0, len(sfens), chunk):
        sl = range(c0, min(c0 + chunk, len(sfens)))
        fi = [[], []]; cnt = [[], []]
        for i in sl:
            s, n = encode_halfka(sfen_state(sfens[i]))
            fi[0] += s; fi[1] += n; cnt[0].append(len(s)); cnt[1].append(len(n))
        TR = []
        for v in (0, 1):
            off = np.concatenate([[0], np.cumsum(cnt[v])[:-1]])
            acc = np.add.reduceat(ftw[np.asarray(fi[v])], off, axis=0, dtype=np.int32) + ftb[None, :]
            acc = np.clip(acc, 0, 254).astype(np.int64)
            TR.append((acc[:, :W // 2] * acc[:, W // 2:]) >> 9)
        TR = np.concatenate(TR, axis=1)                    # [n, W] u8 相当 (stm 側が先)
        bk = buckets[sl.start:sl.stop]
        for b in np.unique(bk):
            m = bk == b
            B0, W0, B1, W1, B2, W2 = net["st"][b]
            l1 = B0[None, :] + TR[m] @ W0.T                 # [k, L1]
            v = l1[:, :M]
            l2in = np.concatenate([np.minimum((v * v) >> 19, 127), np.clip(v >> 6, 0, 127)], axis=1)
            l2 = np.clip((B1[None, :] + l2in @ W1.T) >> 6, 0, 127)
            raw[np.arange(sl.start, sl.stop)[m]] = B2 + l2 @ W2 + l1[:, M]
    return raw


def cdiv(a, d):
    """C の整数除算 (0 方向)"""
    q = np.abs(a) // d
    return np.where(a < 0, -q, q)


def load_m2(n=0):
    R = []
    for i, l in enumerate(io.open(M2, encoding="utf-8")):
        if n and i >= n:
            break
        r = json.loads(l); R.append((r["sfen"], r["bucket"], r["eff_fv"]))
    return R


def main():
    t0 = time.time()
    mode = sys.argv[1]
    if mode == "nagisa":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        R = load_m2(n); net = load_net(NAGISA, nagisa=True)
        s31 = np.array([json.loads(l)["s31"] for _, l in zip(range(len(R)), io.open(M5, encoding="utf-8"))])
        raw = forward(net, [r[0] for r in R], np.array([r[1] for r in R]))
        cp = cdiv(raw, 28)
        d = cp - s31
        print("NAGISA v3.1 検算: %d 局面 / 完全一致 %.4f / |差|<=1 %.4f / |差| 最大 %d / 相関 %.6f / %.0fs" % (
            len(R), (d == 0).mean(), (np.abs(d) <= 1).mean(), np.abs(d).max(), np.corrcoef(cp, s31)[0, 1], time.time() - t0))
        bad = np.flatnonzero(d != 0)[:5]
        for i in bad:
            print("  不一致 #%d bucket %d python %d / exe %d  %s" % (i, R[i][1], cp[i], s31[i], R[i][0]))
        return
    if mode == "check":
        path = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        R = load_m2(n)
    else:
        label, path = sys.argv[2], sys.argv[3]
        out = CACHE % label
        if os.path.exists(out):
            raise SystemExit("🚨キャッシュが既にある (上書きしない): %s" % out)
        R = load_m2()
    net = load_net(path)
    print("net %s / 幅 %d / L1 %d / L2 %d / 読込 %.1fs" % (path, net["W"], net["L1"], net["L2"], time.time() - t0))
    raw = forward(net, [r[0] for r in R], np.array([r[1] for r in R]))
    cp = cdiv(raw, np.array([r[2] for r in R]))
    bk = np.array([r[1] for r in R])
    for name, m in (("bucket 0-2", bk <= 2), ("bucket 3-7", bk >= 3)):
        if m.any():
            print("  %s: n %d / cp 平均 %.1f / sd %.1f / |cp| 中央 %.0f / |cp|>1500 %.4f / 0 の率 %.4f" % (
                name, m.sum(), cp[m].mean(), cp[m].std(), np.median(np.abs(cp[m])), (np.abs(cp[m]) > 1500).mean(), (cp[m] == 0).mean()))
    if not np.isfinite(cp).all() or cp[bk <= 2].std() == 0:
        raise SystemExit("🚨値が壊れている (非有限 / 全部同じ)")
    if mode == "net":
        tmp = out + ".tmp"
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for x in cp:
                f.write(json.dumps(dict(s=int(x))) + "\n")
        os.replace(tmp, out)
        print("書いた %s (%d 行) / %.0fs" % (out, len(cp), time.time() - t0))


if __name__ == "__main__":
    main()
