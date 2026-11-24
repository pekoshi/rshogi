# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""OPN の静的評価 (python 整数順伝播) の拡張版。 2026-11-14 OPNARCH (設計書 10 章)
   runs/invent03/opn_static.py は書き換えず、 その関数 (LEB128 / sfen の復号 / 整数除算) を使う。

■ opn_static.py からの拡張
   1. head の本数を arch から決める: `LSMODE=K9ZP3,` → 243 (k9k9z × 進行度 3) / `LSMODE=K9Z,` → 81 / 無し → 8。
      それ以外の LSMODE= と PSQT / threat / 追加 section は止める。 入力は HalfKA_hm のみ (案 A)。
   2. head 番号: 設計書 2.2 の式。 kz = z9(own(手番側の玉)) * 9 + z9(own(非手番側の玉))、 head = kz * 3 + pb。
      norm / z9 は runs/opnarch/kb_density.py と同じ式 (kb_density.py は import すると全局面を数え始めるので写した)。
      pb は m2 の "bucket" (本番エンジンの progress8kpabs)。 pb >= 3 (序盤帯の外) は pb = 2 の head に寄せる
      (243 の net は序盤帯でしか学習していない。 篩の bucket 3-7 の数字は意味を持たない)。
   3. 符号化を一度だけ行い (encode_states)、 何本の net でも同じ配列で順伝播する (forward_encoded)。 dl_heldout.py が使う。
   4. sfen -> HCP (YaneuraOu の packed sfen 32 byte) の符号器 encode_hcp。 学習器との bucket 突き合わせに使う。

usage:
   py -3 opn_static2.py net    <label> <quantised.bin>        # 篩のキャッシュ screen_<label>.jsonl を書く (既にあれば止める)
   py -3 opn_static2.py check  <quantised.bin> [N=2000]       # 先頭 N 局面だけ計算して分布を出す
   py -3 opn_static2.py same   <label> <quantised.bin>        # 既存の screen_<label>.jsonl を全行一致で再現するか (検証 1)
   py -3 opn_static2.py kbpsv  <out.psv>                      # 篩の bucket 0-2 の局面を PSV に書く (往復の復号も確かめる)
   py -3 opn_static2.py kbcmp  <out.psv> <dump.txt>           # 学習器 (opn_kbucket_dump) の kz / pb8 と python を突き合わせる
   --- 2026-11-14 OPN2048 K16 (玉区分 45 -> 16、 arch `Features=HalfKA_hmK16(Friend)[26064->Wx2]`) ---
   load_net2 は K16 の net も読む (net["k16"])。 符号化 (encode_states) は HalfKA_hm のまま持ち、 forward_encoded が
   K16 の net に渡すときだけ k16_index で写す (dl_heldout.py のキャッシュもそのまま使える)。
   py -3 opn_static2.py k16cmp  <out.psv> <dump.txt>          # 学習器 (opn_k16_check --dump) の K16 添字と python を突き合わせる
   py -3 opn_static2.py k16port <45 区分の quantised.bin> <out.bin>   # FT を K16 に写した net を書く (J0 の移植テスト)
   py -3 opn_static2.py k16diff <label45> <label16>           # 2 つのキャッシュの差を「どちらかの玉が 1〜6 段か」で分けて数える
"""
import io, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "invent03"))
from opn_static import (  # noqa: E402  (stdout の utf-8 化と IDLE 優先度もここで掛かる)
    CACHE, IN_DIM, MAGIC, cdiv, load_m2, pad32, read_block, sfen_state)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from bullet_inference_v19 import encode_halfka, unpack_yo_hcp  # noqa: E402

LSMODE_HEADS = {"K9ZP3": 243, "K9Z": 81}
BAD = ("PSQT=", "Threat", "HandCountDense", "PIN=", "DUAL=", "AUX", "MOB=", "L4=", "EGTMP")


# ---------------------------------------------------------------- king bucket (kb_density.py と同じ式)
def norm(fr, black):                                  # その玉の持ち主から見た座標 (自陣が 7-9 段)
    f, r = fr
    return (f, r) if black else (8 - f, 8 - r)


def z9(f, r):
    return 0 if r < 3 else 1 if r < 6 else 2 if r == 6 else 3 + (r - 7) * 3 + f // 3


def kz_of_state(st):
    """st = sfen_state / unpack_yo_hcp の形。 sq = f*9 + r (f 0 = 1 筋 / r 0 = 1 段)。"""
    bk, wk = st["black_king_sq"], st["white_king_sq"]
    K = (bk // 9, bk % 9); k = (wk // 9, wk % 9)
    black = st["side_to_move"] == 0
    own, opp = (K, k) if black else (k, K)
    return z9(*norm(own, black)) * 9 + z9(*norm(opp, not black))


def heads_of(nheads, kz, pb):
    """head 番号。 8 本の net は進行度 bucket そのもの、 81 は kz、 243 は kz*3 + min(pb, 2)。"""
    kz = np.asarray(kz); pb = np.asarray(pb)
    if nheads == 8:
        return pb.astype(np.int64)
    if nheads == 81:
        return kz.astype(np.int64)
    return (kz * 3 + np.minimum(pb, 2)).astype(np.int64)


# ---------------------------------------------------------------- K16 (2026-11-14 OPN2048、 設計書 DESIGN-20261114-OPN2048-FABLE.md 2.3)
PI = 1629
K16_DIM = 16 * PI                                     # 26,064
# 学習器 shogi_halfka.rs の king_bucket16 と同じ式: kb = f*9 + r、 r >= 6 (7〜9 段) -> f*3 + (r-6)、 それ以外 -> 15
KB16 = np.array([(kb // 9) * 3 + (kb % 9 - 6) if kb % 9 >= 6 else 15 for kb in range(45)], dtype=np.int64)


def k16_index(idx):
    """HalfKA_hm の添字 (kb*1629 + packed) -> K16 の添字 (kb16*1629 + packed)。 配列で受ける。"""
    idx = np.asarray(idx, dtype=np.int64)
    return (KB16[idx // PI] * PI + idx % PI).astype(np.int32)


def k16_port_ftw(ftw45):
    """45 区分の FT 表 [73305, W] (i16) を K16 [26064, W] に写す (設計書 2.4 検証 3 の移植テスト)。
       7〜9 段の 15 区分の行はそのまま、 「その他」(1〜6 段の 30 区分) は 30 行の平均 (0 から遠い側へ丸め)。"""
    W = ftw45.shape[1]
    t = ftw45.reshape(45, PI, W)
    out = np.zeros((16, PI, W), dtype=np.int16)
    for kb in range(45):
        if KB16[kb] < 15:
            out[KB16[kb]] = t[kb]
    other = [kb for kb in range(45) if KB16[kb] == 15]
    assert len(other) == 30
    s = np.zeros((PI, W), dtype=np.int64)
    for kb in other:
        s += t[kb]
    mean = s / 30.0
    out[15] = (np.sign(mean) * np.floor(np.abs(mean) + 0.5)).astype(np.int16)
    return out.reshape(K16_DIM, W)


def leb_encode(v):
    """符号付き LEB128 (学習器 encode_signed_leb128 と同じ) を numpy でまとめて行う。 v は i16 の範囲。"""
    v = np.asarray(v, dtype=np.int64)
    ln = np.where((v >= -64) & (v < 64), 1, np.where((v >= -8192) & (v < 8192), 2, 3))
    assert (v >= -(1 << 20)).all() and (v < (1 << 20)).all()
    off = np.concatenate([[0], np.cumsum(ln)[:-1]])
    out = np.zeros(int(ln.sum()), dtype=np.uint8)
    for k in range(3):
        sel = ln > k
        byte = (v[sel] >> (7 * k)) & 0x7F
        byte = byte | np.where(ln[sel] - 1 > k, 0x80, 0)
        out[off[sel] + k] = byte.astype(np.uint8)
    return out


def leb_block(v):
    data = leb_encode(v)
    return MAGIC + np.uint32(len(data)).tobytes() + data.tobytes()


# ---------------------------------------------------------------- net の読み込み
def load_net2(path):
    """返り値: dict(W, L1, L2, NH, fv, ftb[W] i32, ftw[IN, W] i16, st[h] = (B0, W0, B1, W1, B2, W2), arch)"""
    b = np.fromfile(path, dtype=np.uint8)
    alen = int(b[8:12].view(np.uint32)[0])
    arch = bytes(b[12:12 + alen]).decode("utf-8", "replace")
    p = 12 + alen + 4
    for bad in BAD:
        if bad in arch:
            raise SystemExit("🚨この器は PSQT / threat / 追加 section の無い net 専用: arch に %s がある\n%s" % (bad, arch))
    ms = re.findall(r"LSMODE=([^,]*),", arch)
    if len(ms) > 1 or (ms and ms[0] not in LSMODE_HEADS):
        raise SystemExit("🚨知らない LSMODE: %s (許可は %s)\n%s" % (ms, sorted(LSMODE_HEADS), arch))
    NH = LSMODE_HEADS[ms[0]] if ms else 8
    m = re.search(r"Features=(HalfKA_hm|HalfKA_hmK16)\(Friend\)\[(\d+)->(\d+)x2\]", arch)
    k16 = bool(m) and m.group(1) == "HalfKA_hmK16"
    in_dim = K16_DIM if k16 else IN_DIM
    if not m or int(m.group(2)) != in_dim:
        raise SystemExit("🚨入力は HalfKA_hm (73305) / HalfKA_hmK16 (26064) のみ対応。 arch: %s" % arch)
    W = int(m.group(3))
    L2 = int(re.search(r"AffineTransform\[1<-(\d+)\]", arch).group(1))
    L1 = int(re.search(r"AffineTransform\[(\d+)<-%d\]" % (2 * W), arch).group(1))
    fv = int(re.search(r"fv_scale=(\d+)", arch).group(1))
    ftb, p = read_block(b, p)
    ftw, p = read_block(b, p)
    assert len(ftb) == W and len(ftw) == in_dim * W, "FT の要素数 %d / %d" % (len(ftb), len(ftw))
    assert np.abs(ftw).max() <= 32767
    M = L1 - 1; L2IN = 2 * M
    ls = 4 + 4 * L1 + L1 * pad32(W) + 4 * L2 + L2 * pad32(L2IN) + 4 + pad32(L2)
    rest = len(b) - p
    assert rest == ls * NH, "LayerStack の残り %d が %d x %d と合わない (arch: %s)" % (rest, ls, NH, arch)
    st = []
    for k in range(NH):
        q = p + k * ls + 4
        rd32 = lambda n: (b[q:q + 4 * n].view(np.int32).astype(np.int64), q + 4 * n)
        rd8 = lambda r, c: (b[q:q + r * c].view(np.int8).reshape(r, c).astype(np.int64), q + r * c)
        B0, q = rd32(L1); W0, q = rd8(L1, pad32(W)); B1, q = rd32(L2); W1, q = rd8(L2, pad32(L2IN))
        B2, q = rd32(1); W2, q = rd8(1, pad32(L2))
        st.append((B0, W0[:, :W], B1, W1[:, :L2IN], int(B2[0]), W2[0, :L2]))
    return dict(W=W, L1=L1, L2=L2, NH=NH, fv=fv, ftb=ftb.astype(np.int32),
                ftw=ftw.astype(np.int16).reshape(in_dim, W), st=st, arch=arch, k16=k16)


# ---------------------------------------------------------------- 符号化と順伝播 (p2nnue.cpp と同じ整数演算)
def encode_states(states):
    """HalfKA_hm の添字を全局面ぶん並べる。 返り値 dict(idx[2] (int32 連結), cnt[2] (int32 / 局面))。"""
    idx = [[], []]; cnt = [[], []]
    for st in states:
        s, n = encode_halfka(st)
        idx[0] += s; idx[1] += n; cnt[0].append(len(s)); cnt[1].append(len(n))
    return dict(idx=[np.asarray(v, dtype=np.int32) for v in idx], cnt=[np.asarray(v, dtype=np.int32) for v in cnt])


def forward_encoded(net, enc, heads, chunk=1000):
    W, M = net["W"], net["L1"] - 1
    ftw, ftb = net["ftw"], net["ftb"]
    if net.get("k16"):                                # 符号化は HalfKA_hm のまま持ち、 K16 の net に渡すときだけ写す
        enc = dict(enc, idx=[k16_index(v) for v in enc["idx"]])
    n = len(enc["cnt"][0])
    start = [np.concatenate([[0], np.cumsum(enc["cnt"][v])]) for v in (0, 1)]
    raw = np.zeros(n, dtype=np.int64)
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        TR = []
        for v in (0, 1):
            a, z = start[v][c0], start[v][c1]
            off = start[v][c0:c1] - a
            acc = np.add.reduceat(ftw[enc["idx"][v][a:z]], off, axis=0, dtype=np.int32) + ftb[None, :]
            acc = np.clip(acc, 0, 254).astype(np.int64)
            TR.append((acc[:, :W // 2] * acc[:, W // 2:]) >> 9)
        TR = np.concatenate(TR, axis=1)
        hk = heads[c0:c1]
        for h in np.unique(hk):
            m = hk == h
            B0, W0, B1, W1, B2, W2 = net["st"][h]
            l1 = B0[None, :] + TR[m] @ W0.T
            v = l1[:, :M]
            l2in = np.concatenate([np.minimum((v * v) >> 19, 127), np.clip(v >> 6, 0, 127)], axis=1)
            l2 = np.clip((B1[None, :] + l2in @ W1.T) >> 6, 0, 127)
            raw[np.arange(c0, c1)[m]] = B2 + l2 @ W2 + l1[:, M]
    return raw


# ---------------------------------------------------------------- sfen -> HCP (YaneuraOu の packed sfen、 unpack_yo_hcp の逆)
HUF = {1: (0x01, 2), 2: (0x03, 4), 3: (0x0b, 4), 4: (0x07, 4), 5: (0x1f, 6), 6: (0x3f, 6), 7: (0x0f, 5)}
UNPROMO = {9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
FULL = {1: 18, 2: 4, 3: 4, 4: 4, 5: 2, 6: 2, 7: 4}


def encode_hcp(st):
    bits = []
    put = lambda v, n: bits.extend((v >> i) & 1 for i in range(n))
    put(st["side_to_move"], 1); put(st["black_king_sq"], 7); put(st["white_king_sq"], 7)
    used = {pt: 0 for pt in FULL}
    for sq, (c, pt) in enumerate(st["board"]):
        if sq in (st["black_king_sq"], st["white_king_sq"]):
            continue
        if pt == 0:
            put(0, 1); continue
        base = UNPROMO.get(pt, pt); used[base] += 1
        code, n = HUF[base]; put(code, n)
        if base != 7:
            put(1 if pt in UNPROMO else 0, 1)
        put(c, 1)
    for c, hand in ((0, st["black_hand"]), (1, st["white_hand"])):
        for pt in (1, 2, 3, 4, 7, 5, 6):
            for _ in range(hand.get(pt, 0)):
                used[pt] += 1
                code, n = HUF[pt]; put(code >> 1, n - 1)
                if pt != 7:
                    put(0, 1)
                put(c, 1)
    for pt in (1, 2, 3, 4, 7, 5, 6):                    # 駒箱 (盤にも手駒にも無い駒) で 256 bit を埋める
        for _ in range(FULL[pt] - used[pt]):
            code, n = HUF[pt]; put(code >> 1, n - 1)
            if pt != 7:
                put(1, 1)
            put(0, 1)
    assert len(bits) == 256, "HCP が 256 bit にならない (%d)" % len(bits)
    out = bytearray(32)
    for i, v in enumerate(bits):
        out[i >> 3] |= v << (i & 7)
    return bytes(out)


def same_state(a, b):
    keys = ("side_to_move", "black_king_sq", "white_king_sq")
    if any(a[k] != b[k] for k in keys) or list(a["board"]) != list(b["board"]):
        return False
    hz = lambda h: {k: v for k, v in h.items() if v}
    return hz(a["black_hand"]) == hz(b["black_hand"]) and hz(a["white_hand"]) == hz(b["white_hand"])


# ---------------------------------------------------------------- main
def m2_rows():
    return [json.loads(l) for l in io.open("E:/shogi/p2shogi/tmp_mob/v4evalana/m2_games.jsonl", encoding="utf-8")]


def eval_m2(net, R):
    states = [sfen_state(r[0]) for r in R]
    pb = np.array([r[1] for r in R])
    kz = np.array([kz_of_state(s) for s in states])
    raw = forward_encoded(net, encode_states(states), heads_of(net["NH"], kz, pb))
    return cdiv(raw, np.array([r[2] for r in R])), pb


def main():
    t0 = time.time()
    mode = sys.argv[1]
    if mode in ("net", "check", "same"):
        if mode == "check":
            path = sys.argv[2]; R = load_m2(int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
        else:
            label, path = sys.argv[2], sys.argv[3]; R = load_m2()
            if mode == "net" and os.path.exists(CACHE % label):
                raise SystemExit("🚨キャッシュが既にある (上書きしない): %s" % (CACHE % label))
        net = load_net2(path)
        print("net %s / 幅 %d / L1 %d / L2 %d / head %d / fv_scale %d / 読込 %.1fs" % (
            path, net["W"], net["L1"], net["L2"], net["NH"], net["fv"], time.time() - t0))
        cp, bk = eval_m2(net, R)
        for name, m in (("bucket 0-2", bk <= 2), ("bucket 3-7", bk >= 3)):
            if m.any():
                print("  %s: n %d / cp 平均 %.1f / sd %.1f / |cp| 中央 %.0f / |cp|>1500 %.4f / 0 の率 %.4f" % (
                    name, m.sum(), cp[m].mean(), cp[m].std(), np.median(np.abs(cp[m])), (np.abs(cp[m]) > 1500).mean(), (cp[m] == 0).mean()))
        if mode == "same":
            old = np.array([json.loads(l)["s"] for l in io.open(CACHE % label, encoding="utf-8")])
            assert len(old) == len(cp), "行数が違う %d / %d" % (len(old), len(cp))
            print("既存キャッシュ %s との一致: 全行一致 %s / 不一致 %d 行 / %.0fs" % (
                CACHE % label, bool((old == cp).all()), int((old != cp).sum()), time.time() - t0))
            return
        if not np.isfinite(cp).all() or cp[bk <= 2].std() == 0:
            raise SystemExit("🚨値が壊れている (非有限 / 全部同じ)")
        if mode == "net":
            out = CACHE % label; tmp = out + ".tmp"
            with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
                for x in cp:
                    f.write(json.dumps(dict(s=int(x))) + "\n")
            os.replace(tmp, out)
            print("書いた %s (%d 行) / %.0fs" % (out, len(cp), time.time() - t0))
        return
    if mode == "kbpsv":
        out = sys.argv[2]
        R = [r for r in m2_rows() if r["bucket"] <= 2]
        bad = 0
        with open(out + ".tmp", "wb") as f:
            for r in R:
                st = sfen_state(r["sfen"])
                h = encode_hcp(st)
                if not same_state(unpack_yo_hcp(h), st):
                    bad += 1
                f.write(h + np.int16(0).tobytes() + np.uint16(0).tobytes() + np.uint16(r["ply"]).tobytes() + b"\x00\x00")
        os.replace(out + ".tmp", out)
        print("篩の bucket 0-2 %d 局面を PSV に書いた -> %s / HCP の往復 (encode -> unpack_yo_hcp) の食い違い %d / %.0fs" % (
            len(R), out, bad, time.time() - t0))
        return
    if mode == "kbcmp":
        psv, dump = sys.argv[2], sys.argv[3]
        R = [r for r in m2_rows() if r["bucket"] <= 2]
        D = np.loadtxt(dump, dtype=np.int64).reshape(-1, 3)
        assert len(D) == len(R) == os.path.getsize(psv) // 40, "件数が合わない %d / %d" % (len(D), len(R))
        kz_py = np.array([kz_of_state(sfen_state(r["sfen"])) for r in R])
        pb_m2 = np.array([r["bucket"] for r in R])
        head_py = heads_of(243, kz_py, pb_m2)
        print("学習器 (opn_kbucket_dump) と python の突き合わせ: 篩の bucket 0-2 の %d 局面" % len(R))
        print("  kz (k9k9z)        一致 %d / 不一致 %d" % ((D[:, 0] == kz_py).sum(), (D[:, 0] != kz_py).sum()))
        print("  進行度 bucket      学習器 pb8 と m2 (本番エンジン) の一致 %d / 不一致 %d" % ((D[:, 1] == pb_m2).sum(), (D[:, 1] != pb_m2).sum()))
        print("  head (kz*3+pb)    一致 %d / 不一致 %d / 使われた head %d" % (
            (D[:, 2] == head_py).sum(), (D[:, 2] != head_py).sum(), len(np.unique(head_py))))
        for i in np.flatnonzero((D[:, 0] != kz_py) | (D[:, 2] != head_py))[:5]:
            print("    不一致 #%d 学習器 %s / python kz %d head %d  %s" % (i, D[i].tolist(), kz_py[i], head_py[i], R[i]["sfen"]))
        return
    if mode == "k16cmp":
        # 学習器 (opn_k16_check --dump) の K16 添字の集合 hash と、 python の符号化 (encode_halfka -> k16_index) を突き合わせる
        psv, dump = sys.argv[2], sys.argv[3]
        R = [r for r in m2_rows() if r["bucket"] <= 2]
        D = np.loadtxt(dump, dtype=np.uint64).reshape(-1, 4)
        assert len(D) == len(R) == os.path.getsize(psv) // 40, "件数が合わない %d / %d" % (len(D), len(R))
        HM = (1 << 61) - 1
        bad = 0; n_other = 0
        for i, r in enumerate(R):
            s, n = encode_halfka(sfen_state(r["sfen"]))
            row = []
            for v in (s, n):
                h = 0
                for x in sorted(k16_index(v).tolist()):
                    h = (h * 1000003 + x + 1) % HM
                row += [len(v), h]
            n_other += int(k16_index(s[:1])[0] >= 15 * PI) + int(k16_index(n[:1])[0] >= 15 * PI)
            if [int(x) for x in D[i]] != row:
                bad += 1
                if bad <= 5:
                    print("    不一致 #%d 学習器 %s / python %s  %s" % (i, D[i].tolist(), row, r["sfen"]))
        print("学習器 (opn_k16_check) と python の K16 添字の突き合わせ: 篩の bucket 0-2 の %d 局面 / 一致 %d / 不一致 %d / "
              "「その他」区分の視点 %d / %.0fs" % (len(R), len(R) - bad, bad, n_other, time.time() - t0))
        if bad:
            raise SystemExit("🚨K16 の添字が学習器と合わない")
        return
    if mode == "k16port":
        # 45 区分の net (HalfKA_hm、 PSQT なし) の FT を K16 に写した quantised.bin を書く。 後段 (LayerStack) はそのまま。
        src, dst = sys.argv[2], sys.argv[3]
        if os.path.exists(dst):
            raise SystemExit("🚨出力が既にある (上書きしない): %s" % dst)
        net = load_net2(src)
        if net["k16"]:
            raise SystemExit("🚨元の net が既に K16")
        b = np.fromfile(src, dtype=np.uint8)
        alen = int(b[8:12].view(np.uint32)[0])
        arch = bytes(b[12:12 + alen]).decode("utf-8")
        old = "Features=HalfKA_hm(Friend)[%d->" % IN_DIM
        assert arch.count(old) == 1, arch
        arch16 = arch.replace(old, "Features=HalfKA_hmK16(Friend)[%d->" % K16_DIM).encode("utf-8")
        p = 12 + alen + 4
        _, p_w = read_block(b, p)                    # biases の後ろ = weights の先頭
        _, p_end = read_block(b, p_w)
        # 符号器の検算: 元の weights を python で符号化し直して元のバイト列と完全一致するか
        if leb_block(net["ftw"].reshape(-1)) != bytes(b[p_w:p_end]):
            raise SystemExit("🚨LEB128 の符号器が学習器と合わない (元の weights を再現できない)")
        ftw16 = k16_port_ftw(net["ftw"])
        out = (bytes(b[:8]) + np.uint32(len(arch16)).tobytes() + arch16 + bytes(b[12 + alen:p_w])
               + leb_block(ftw16.reshape(-1)) + bytes(b[p_end:]))
        with open(dst + ".tmp", "wb") as f:
            f.write(out)
        os.replace(dst + ".tmp", dst)
        chk = load_net2(dst)
        ok = (chk["k16"] and (chk["ftw"] == ftw16).all() and (chk["ftb"] == net["ftb"]).all()
              and all(all(np.array_equal(x, y) for x, y in zip(a, c)) for a, c in zip(net["st"], chk["st"])))
        print("K16 に写した net -> %s (%d byte) / 符号器の検算 一致 / 読み戻し %s / arch %s / %.0fs" % (
            dst, len(out), "一致" if ok else "🚨不一致", chk["arch"], time.time() - t0))
        if not ok:
            raise SystemExit("🚨書いた net の読み戻しが合わない")
        return
    if mode == "k16diff":
        # 2 つのキャッシュ (45 区分の net / その K16 移植) の値の差を「どちらかの玉が 1〜6 段か」で分けて数える
        la, lb = sys.argv[2], sys.argv[3]
        R = m2_rows()
        A = np.array([json.loads(l)["s"] for l in io.open(CACHE % la, encoding="utf-8")])
        B = np.array([json.loads(l)["s"] for l in io.open(CACHE % lb, encoding="utf-8")])
        assert len(A) == len(B) == len(R)
        oth = np.zeros(len(R), dtype=bool)
        for i, r in enumerate(R):
            s, n = encode_halfka(sfen_state(r["sfen"]))
            oth[i] = bool(s) and (k16_index(s[:1])[0] >= 15 * PI or k16_index(n[:1])[0] >= 15 * PI)
        bk = np.array([r["bucket"] for r in R])
        for name, m in (("全局面", bk >= 0), ("bucket 0-1", bk <= 1), ("bucket 0-2", bk <= 2)):
            for g, gm in (("両玉とも 7〜9 段", ~oth), ("どちらかが 1〜6 段", oth)):
                mm = m & gm; d = A[mm] != B[mm]
                print("  %-10s %-18s n %6d / 値が違う %6d / |差| 平均 %.1f 最大 %d" % (
                    name, g, mm.sum(), d.sum(), np.abs(A[mm] - B[mm]).mean() if mm.any() else 0, np.abs(A[mm] - B[mm]).max() if mm.any() else 0))
        if (A[~oth] != B[~oth]).any():
            raise SystemExit("🚨両玉とも 7〜9 段の局面で値が違う (写しか添字の誤り)")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
