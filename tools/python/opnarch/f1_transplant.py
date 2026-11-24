# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""試験 F1 の準備: 本体 (現職 top3q-102) の FT を、 幅 1024 の OPN (HalfKA_hm 全表 + factoriser、 学習器 shogi_layerstack_BFS.exe)
   の初期重みに写す。 設計書 runs/prereg/DESIGN-20261114-OPN2048-FABLE.md 5.1 節 / 8 章 F1。 2026-11-14
   CPU だけ (GPU は使わない)。 大きな配列は memmap と塊ごとの処理でメモリを抑える。

■ 写し方
   - 本体の f32 checkpoint (top3q-102/optimiser_state/weights.bin) の l0w [73,305 x 1024] / l0b [1024] をそのまま使う
     (量子化済みの表から戻す必要はない。 src で「quantised.bin の FT = round(254 * f32)」を全要素で確かめる)。
   - OPN の l0w は [1,629 行の factoriser | 73,305 行の実特徴] (行 = 特徴、 1 行 = 1024 列が連続)。 factoriser 行は 0、
     実特徴の行に本体の l0w を置く -> 学習器の書き出し (実特徴 + factor) は本体の FT と bit 単位で同じになる。
   - 列: 本体も OPN も幅 1024 で「列 i と列 i + 512 の要素積」(clip 0..254、 >>9)。 同じ幅なので列は並べ替えない (i -> i)。
   - 後段 (l1w / l1b / l1fw / l1fb / l2w / l2b / l3w / l3b) は weights.bin に**入れない**。 bullet のローダはファイル側の
     項目だけを上書きするので (acyclib graph.rs load_from_file_adapted)、 無い項目は学習器自身の初期化
     (l1w / l1b = 0、 l1fw = N(0, sqrt(2/1024))、 l2w = N(0, sqrt(2/30))、 l3w = N(0, sqrt(2/32))、 bias = 0) のまま始まる。
     = 「学習器の初期化と同じ分布」を学習器そのものに任せる。 PSQT / threat は OPN に無いので写さない。

usage:
   py -3 f1_transplant.py src                 # 本体の f32 と quantised.bin の FT の一致 + PSQT による HalfKA 添字の照合 (engine の so_psqt)
   py -3 f1_transplant.py build  <out_dir>    # <out_dir>/optimiser_state/weights.bin (l0w, l0b の 2 項目) を書く
   py -3 f1_transplant.py verify <out_dir> [N=20000]   # 書いた重みを学習器の式で量子化 -> 本体の表と全要素一致 + アキュムレータ一致
"""
import io, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "opnarch"))
from opn_static2 import read_block, sfen_state, encode_states  # noqa: E402  (IDLE 優先度 / utf-8 もここで掛かる)

SRC = "E:/shogi/p2shogi/tmp_mob/top3q/top3q-102"            # 現職 (本番 toml の EvalFile = _kept_nets/top3q-102.bin と md5 一致)
REF_OPN = "E:/shogi/p2shogi/tmp_mob/opnW1024x/opnW1024x-8"    # 並びの見本 (幅 1024 の OPN、 同じ学習器)
M2 = "E:/shogi/p2shogi/tmp_mob/v4evalana/m2_games.jsonl"
IN, PI, W, NB = 73305, 1629, 1024, 8
QA_FT = 254.0


def index_weights(path):
    """weights.bin の目次: {id: (要素数, f32 の先頭 byte 位置)}。 中身は読まない。"""
    out = {}; sz = os.path.getsize(path)
    with open(path, "rb") as f:
        p = 0
        while p < sz:
            f.seek(p); head = f.read(256)
            e = head.index(b"\n"); wid = head[:e].decode("ascii")
            n = int(np.frombuffer(head[e + 1:e + 9], "<u8")[0])
            out[wid] = (n, p + e + 9); p = p + e + 9 + 4 * n
    return out


def tensor(path, idx, wid):
    n, off = idx[wid]
    return np.memmap(path, dtype="<f4", mode="r", offset=off, shape=(n,))


def rnd(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)                # Rust の f64::round (0 から遠い側)


def q16(x32):
    """学習器の FT 書き出し: (254 * v as f64).round() as i16 (飽和)。"""
    return np.clip(rnd(QA_FT * x32.astype(np.float64)), -32768, 32767).astype(np.int16)


def main_quant_ft():
    """本体 quantised.bin の FT (ftb [W], ftw [IN, W]) と PSQT (bias [8], w [IN, 8]) を読む。"""
    b = np.fromfile(os.path.join(SRC, "quantised.bin"), dtype=np.uint8)
    alen = int(b[8:12].view(np.uint32)[0])
    arch = bytes(b[12:12 + alen]).decode("utf-8", "replace")
    assert "Features=HalfKA_hm(Friend)[73305->1024x2]" in arch and "PSQT=8" in arch, arch
    p = 12 + alen + 4
    ftb, p = read_block(b, p)
    ftw, p = read_block(b, p)
    assert len(ftb) == W and len(ftw) == IN * W
    ps = b[p:p + 4 * NB * (IN + 1)].view(np.int32)
    return arch, ftb.astype(np.int16), ftw.astype(np.int16).reshape(IN, W), ps[:NB].astype(np.int64), ps[NB:].reshape(IN, NB).astype(np.int64)


def m2_rows(n):
    R = []
    for ln in io.open(M2, encoding="utf-8"):
        r = json.loads(ln)
        if r["bucket"] <= 2:
            R.append(r)
            if len(R) >= n:
                break
    return R


def cmd_src():
    t0 = time.time()
    wpath = os.path.join(SRC, "optimiser_state", "weights.bin")
    idx = index_weights(wpath)
    print("本体 %s の weights.bin: %s" % (SRC, {k: v[0] for k, v in idx.items()}))
    assert idx["l0w"][0] == IN * W and idx["l0b"][0] == W, "本体の l0w / l0b の形が想定 (73305 x 1024、 factoriser なし) と違う"
    arch, ftb, ftw, pb, pw = main_quant_ft()
    print("arch: %s" % arch)
    l0w = tensor(wpath, idx, "l0w"); l0b = tensor(wpath, idx, "l0b")
    bad = int((q16(np.asarray(l0b)) != ftb).sum()); CH = 4096
    for r0 in range(0, IN, CH):
        r1 = min(IN, r0 + CH)
        bad += int((q16(np.asarray(l0w[r0 * W:r1 * W])).reshape(-1, W) != ftw[r0:r1]).sum())
    print("FT: quantised.bin と round(254 * f32) の食い違い %d / %d 要素 (0 なら f32 がそのまま元の表)" % (bad, IN * W + W))
    zc = np.flatnonzero((ftw == 0).all(axis=0))
    print("FT: 全行 0 の列 %d 本 %s (threat 専用列があれば出る) / |w| 最大 %d / 0 の率 %.4f / bias 範囲 [%d, %d]" % (
        len(zc), zc[:10].tolist(), int(np.abs(ftw).max()), float((ftw == 0).mean()), int(ftb.min()), int(ftb.max())))
    # HalfKA 添字の照合: engine (本体 exe) の so_psqt を python の添字 (encode_halfka = OPN の器と同じ) で再現できるか
    R = m2_rows(5000)
    states = [sfen_state(r["sfen"]) for r in R]
    enc = encode_states(states)
    bk = np.array([r["bucket"] for r in R]); so_psqt = np.array([r["so_psqt"] for r in R])
    st = [np.concatenate([[0], np.cumsum(enc["cnt"][v])]) for v in (0, 1)]
    P = [np.add.reduceat(pw[enc["idx"][v]], st[v][:-1], axis=0) + pb[None, :] for v in (0, 1)]
    ps, pn = P[0][np.arange(len(R)), bk], P[1][np.arange(len(R)), bk]
    for name, val in (("(stm - ntm) / 2", (ps - pn) // 2), ("(stm - ntm) / 2 (0 方向)", np.trunc((ps - pn) / 2).astype(np.int64)),
                      ("stm - ntm", ps - pn)):
        print("PSQT の照合 %-24s: 一致 %d / %d 局面 (|差| 最大 %d)" % (name, int((val == so_psqt).sum()), len(R), int(np.abs(val - so_psqt).max())))
    if bad:
        raise SystemExit("🚨本体の quantised.bin の FT が f32 と合わない")
    print("%.0fs" % (time.time() - t0))


def cmd_build(out_dir):
    t0 = time.time()
    od = os.path.join(out_dir, "optimiser_state"); dst = os.path.join(od, "weights.bin")
    if os.path.exists(dst):
        raise SystemExit("🚨出力が既にある (上書きしない): %s" % dst)
    ridx = index_weights(os.path.join(REF_OPN, "optimiser_state", "weights.bin"))
    assert ridx["l0w"][0] == (PI + IN) * W and ridx["l0b"][0] == W, "見本 OPN の l0w / l0b の形が想定と違う: %s" % ridx
    print("見本 OPN %s の項目: %s" % (REF_OPN, {k: v[0] for k, v in ridx.items()}))
    wpath = os.path.join(SRC, "optimiser_state", "weights.bin"); idx = index_weights(wpath)
    l0w = tensor(wpath, idx, "l0w"); l0b = tensor(wpath, idx, "l0b")
    os.makedirs(od, exist_ok=True)
    with open(dst + ".tmp", "wb") as f:
        f.write(b"l0w\n" + np.uint64((PI + IN) * W).tobytes())
        f.write(np.zeros(PI * W, dtype="<f4").tobytes())             # factoriser 1,629 行 = 0
        CH = 8192
        for r0 in range(0, IN, CH):
            r1 = min(IN, r0 + CH)
            f.write(np.asarray(l0w[r0 * W:r1 * W], dtype="<f4").tobytes())
        f.write(b"l0b\n" + np.uint64(W).tobytes() + np.asarray(l0b, dtype="<f4").tobytes())
    os.replace(dst + ".tmp", dst)
    with open(os.path.join(out_dir, "SOURCE.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("F1 初期重み: l0w = [factoriser 1629 行 = 0 | %s の l0w 73305 行] / l0b = 同 l0b / 後段は学習器の初期化 (weights.bin に入れない)\n" % wpath)
    print("書いた %s (%d byte、 項目 l0w %d / l0b %d) / %.0fs" % (dst, os.path.getsize(dst), (PI + IN) * W, W, time.time() - t0))


def cmd_verify(out_dir, n):
    t0 = time.time()
    dst = os.path.join(out_dir, "optimiser_state", "weights.bin")
    didx = index_weights(dst)
    ridx = index_weights(os.path.join(REF_OPN, "optimiser_state", "weights.bin"))
    print("書いた weights.bin の項目: %s" % {k: v[0] for k, v in didx.items()})
    for k in didx:
        assert k in ridx and didx[k][0] == ridx[k][0], "項目 %s の要素数が見本 OPN と違う" % k
    _, ftb, ftw, _, _ = main_quant_ft()
    lw = tensor(dst, didx, "l0w"); lb = tensor(dst, didx, "l0b")
    fac = np.asarray(lw[:PI * W]).reshape(PI, W)
    print("factoriser 行: 0 以外の要素 %d" % int((fac != 0).sum()))
    # 学習器の書き出し (shogi_layerstack.rs: coalesced = l0w[1629 + r] + l0w[r % 1629] を f32 で足して round(254 *)) を再現
    q = np.empty((IN, W), dtype=np.int16); CH = 4096
    for r0 in range(0, IN, CH):
        r1 = min(IN, r0 + CH)
        real = np.asarray(lw[(PI + r0) * W:(PI + r1) * W]).reshape(-1, W)
        q[r0:r1] = q16(real + fac[np.arange(r0, r1) % PI])
    qb = q16(np.asarray(lb))
    print("量子化した FT と本体 quantised.bin の FT: 重みの食い違い %d / %d、 bias の食い違い %d / %d" % (
        int((q != ftw).sum()), q.size, int((qb != ftb).sum()), W))
    # アキュムレータ (clip 前の i32) を篩の bucket 0-2 の局面で: OPN の器 (opn_static2 の forward と同じ和) / 本体の表
    R = m2_rows(n)
    enc = encode_states([sfen_state(r["sfen"]) for r in R])
    st = [np.concatenate([[0], np.cumsum(enc["cnt"][v])]) for v in (0, 1)]
    worst = 0; act = []
    for v in (0, 1):
        for c0 in range(0, len(R), 1000):
            c1 = min(c0 + 1000, len(R)); a, z = st[v][c0], st[v][c1]; off = st[v][c0:c1] - a
            ao = np.add.reduceat(q[enc["idx"][v][a:z]], off, axis=0, dtype=np.int32) + qb[None, :].astype(np.int32)
            am = np.add.reduceat(ftw[enc["idx"][v][a:z]], off, axis=0, dtype=np.int32) + ftb[None, :].astype(np.int32)
            worst = max(worst, int(np.abs(ao - am).max()))
            act.append(np.clip(ao, 0, 254))
    A = np.concatenate(act)
    print("アキュムレータ (%d 局面 x 2 視点 x %d): OPN 側 - 本体 の |差| 最大 %d / clip 後 0 の率 %.3f / 254 の率 %.3f / 中央 %.0f" % (
        len(R), W, worst, float((A == 0).mean()), float((A == 254).mean()), float(np.median(A))))
    if (q != ftw).any() or (qb != ftb).any() or worst or (fac != 0).any():
        raise SystemExit("🚨写しが本体の FT と一致しない")
    print("OK / %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else ""
    if m == "src":
        cmd_src()
    elif m == "build":
        cmd_build(sys.argv[2])
    elif m == "verify":
        cmd_verify(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
    else:
        raise SystemExit(__doc__)
