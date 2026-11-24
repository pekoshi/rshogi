# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""FC1PREC 段 3 (a) OPN: python の整数模擬とエンジンの OPN の静的評価 (opn_raw) の完全一致
(runs/fti32_20261118/fti32_verify_opn.py の写し)。 ついでに同じ局面で OPN の整数 − f32 の差を新旧の式で出す (OPN は threat / KH が無いので python の値が本当の値)。

  py -3 fc1p_verify_opn.py <exe (P2_DIAG + P2_FC1_PREC)> <本体 net (新形式)> <OPN net (新形式)> <OPN 旧 net (FTI32)> <OPN weights.bin> <m2_games.jsonl> <out.txt> [N=1000]

head (fc 層) は net のファイルの末尾 8 head から読む (新形式 / 旧形式とも)。 FT は weights.bin から k=8 の式 (書き出し器 fti32_write.py --opn と同じ)。
新の式: l2in = [min(floor(v²/2^17), 508), clip(floor(v/2^4), 0, 508)] / l2 = clip(floor((b1 + w1·l2in) / 2^7), 0, 508) / raw = floor((b2 + w2·l2 + 2) / 4) + l1[skip]
旧の式: l2in = [min(floor(v²/2^19), 127), clip(floor(v/2^6), 0, 127)] / l2 = clip(floor((b1 + w1·l2in) / 2^6), 0, 127) / raw = b2 + w2·l2 + l1[skip]
f32: FT も後段も連続 (s0_opn.py の f32 と同じ式)。 cp は /28。
"""
import os
import io, json, re, struct, subprocess, sys
import numpy as np
import cshogi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k256_20261117"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "opnarch"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blunders_20261116"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4confirm"))
from bullet_inference_v19 import unpack_yo_hcp, encode_halfka  # noqa: E402
from k256_merge import read_weight  # noqa: E402
from opn_static2 import k16_index  # noqa: E402
import eng, v4c  # noqa: E402

exe, mnet, onet, oold, owts, jl, outp = sys.argv[1:8]
N = int(sys.argv[8]) if len(sys.argv) > 8 else 1000
K, MM, NB, W, L1, L2, PI = 8, 32, 8, 2048, 16, 32, 1629
M, HALF, FIN = L1 - 1, W // 2, W   # W = FT 幅、 fc0 の入力 = 2 視点 x W/2


def heads(path, want_ver):
    fb = open(path, "rb").read()
    assert struct.unpack_from("<I", fb, 0)[0] == want_ver, path
    bs = 4 + 4 * L1 + L1 * FIN + 4 * L2 + L2 * 32 + 4 + 32
    st = len(fb) - NB * bs
    H = []
    for k in range(NB):
        p = st + k * bs + 4
        b0 = np.frombuffer(fb, "<i4", L1, p).astype(np.float64); p += 4 * L1
        w0 = np.frombuffer(fb, "i1", L1 * FIN, p).reshape(L1, FIN).astype(np.float64); p += L1 * FIN
        b1 = np.frombuffer(fb, "<i4", L2, p).astype(np.float64); p += 4 * L2
        w1 = np.frombuffer(fb, "i1", L2 * 32, p).reshape(L2, 32)[:, :2 * M].astype(np.float64); p += L2 * 32
        b2 = float(struct.unpack_from("<i", fb, p)[0]); p += 4
        w2 = np.frombuffer(fb, "i1", L2, p).astype(np.float64)
        H.append((b0, w0, b1, w1, b2, w2))
    return H


HN, HO = heads(onet, 0x7AF32F22), heads(oold, 0x7AF32F21)
T = {n: read_weight(owts, n) for n in ("l0w", "l0b", "l1w", "l1fw", "l1b", "l1fb", "l2w", "l2b", "l3w", "l3b")}
l0 = T["l0w"].reshape(-1, W)
ftw32 = (l0[PI:].reshape(16, PI, W) + l0[:PI][None, :, :]).reshape(16 * PI, W)
del l0
FTW = np.round(np.float32(254.0 * K) * ftw32)
FTB = np.round(254.0 * K * T["l0b"].astype(np.float64))
FTWF = np.float32(254.0) * ftw32; FTBF = 254.0 * T["l0b"].astype(np.float64)
del ftw32
merged = T["l1w"].reshape(FIN, NB, L1).astype(np.float64) + T["l1fw"].reshape(FIN, L1)[:, None, :]
biasf = T["l1b"].reshape(NB, L1).astype(np.float64) + T["l1fb"][None, :]
l2wf = T["l2w"].reshape(2 * M, NB, L2).astype(np.float64); l2b = T["l2b"].astype(np.float64)
l3wf = T["l3w"].reshape(L2, NB).astype(np.float64); l3b = T["l3b"].astype(np.float64)
del T


def feats_of(sfen):
    board = cshogi.Board(sfen); buf = np.zeros(32, np.uint8); board.to_psfen(buf)
    fs, fn = encode_halfka(unpack_yo_hcp(bytes(buf)))
    return [k16_index(f) for f in (fs, fn)]


def py_int(F, h, new):
    A = [np.clip(FTB + FTW[f].sum(0, dtype=np.float64), 0, 254 * K) for f in F]
    dv = 512.0 * K * K / MM
    TR = np.r_[np.floor(A[0][:HALF] * A[0][HALF:] / dv), np.floor(A[1][:HALF] * A[1][HALF:] / dv)]
    b0, w0, b1, w1, b2, w2 = (HN if new else HO)[h]
    l1 = np.floor((MM * b0 + w0 @ TR) / MM); v = l1[:M]
    if new:
        l2in = np.r_[np.minimum(np.floor(v * v / 2 ** 17), 508), np.clip(np.floor(v / 16), 0, 508)]
        l2 = np.clip(np.floor((b1 + w1 @ l2in) / 128), 0, 508)
        return int(np.floor((b2 + l2 @ w2 + 2) / 4) + l1[M])
    l2in = np.r_[np.minimum(np.floor(v * v / 2 ** 19), 127), np.clip(np.floor(v / 64), 0, 127)]
    l2 = np.clip(np.floor((b1 + w1 @ l2in) / 64), 0, 127)
    return int(b2 + l2 @ w2 + l1[M])


def py_f32(F, h):
    A = [np.clip(FTBF + FTWF[f].sum(0, dtype=np.float64), 0, 254) for f in F]
    TR = np.r_[A[0][:HALF] * A[0][HALF:] / 512.0, A[1][:HALF] * A[1][HALF:] / 512.0]
    l1 = 8128.0 * biasf[h] + (64.0 * merged[:, h, :].T) @ TR; v = l1[:M]
    l2in = np.r_[np.clip(v * v / 2 ** 19, 0, 127), np.clip(v / 64, 0, 127)]
    l2 = np.clip(127.0 * l2b[h * L2:(h + 1) * L2] + l2wf[:, h, :].T @ l2in, 0, 127)
    return 8128.0 * l3b[h] + l2 @ (64.0 * l3wf[:, h]) + l1[M]


J = json.load(io.open("C:/project/p2shogi/runs/v4confirm/it09_capsee.json", encoding="utf-8"))
o, _ = v4c.our_options(J)
OPTS = [(k, v) for k, v in o if k not in set(eng.DROPK) | {"EvalFile", "KHFile", "OPNFile"}] + [
    ("EvalFile", mnet), ("KHFile", "E:/shogi/p2shogi/tmp_mob/khhr/kh_x125.bin"), ("OPNFile", onet), ("USI_OwnBook", "false"),
    ("Threads", "1"), ("USI_Hash", "16")]
R = [json.loads(l) for l in io.open(jl, encoding="utf-8")]
rng = np.random.default_rng(20261118)
idx = []
for b in (0, 1):
    idx += list(rng.choice([i for i, r in enumerate(R) if r["bucket"] == b and not r.get("incheck")], size=N // 2, replace=False))
p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     encoding="utf-8", errors="replace", bufsize=1)


def send(s):
    p.stdin.write(s + "\n"); p.stdin.flush()


def until(pref):
    ls = []
    while True:
        l = p.stdout.readline()
        if not l:
            raise SystemExit("エンジンが落ちた:\n" + "".join(ls[-20:]))
        ls.append(l)
        if l.startswith(pref) or (pref.startswith('info string ') and pref in l):
            return ls


send("usi"); until("usiok")
send("\n".join("setoption name %s value %s" % kv for kv in OPTS))
send("isready"); head = until("readyok")
info = [l.strip() for l in head if "FT k=" in l or "FC1PREC" in l or ("OPN" in l and ("loaded" in l or "失敗" in l or "error" in l.lower()))]
bad = []; n = 0; pbs = {}; EN, EO, PB = [], [], []
for i in idx:
    sfen = R[i]["sfen"]
    send("position sfen " + sfen)
    send("eval")
    ls = until("info string P2OPN pb=")
    m = [re.search(r"P2OPN pb=(\d+) .*?opn_raw=(-?\d+)", l) for l in ls]
    m = [x for x in m if x]
    if len(m) != 1:
        raise SystemExit("P2OPN の行が取れない:\n" + "".join(ls[-10:]))
    pb, er = int(m[0].group(1)), int(m[0].group(2))
    F = feats_of(sfen)
    pr = py_int(F, pb, True)
    f = py_f32(F, pb)
    EN.append((pr - f) / 28.0); EO.append((py_int(F, pb, False) - f) / 28.0); PB.append(pb)
    pbs[pb] = pbs.get(pb, 0) + 1
    n += 1
    if pr != er:
        bad.append((sfen, pb, er, pr))
send("quit")
p.wait(timeout=60)
EN, EO, PB = np.array(EN), np.array(EO), np.array(PB)
out = [f"FC1PREC 検証 (a) OPN exe {exe}", f"OPN net {onet} / 局面 {n} (pb 分布 {dict(sorted(pbs.items()))}) / 不一致 {len(bad)}"] + info
for s, pb, er, pr in bad[:5]:
    out.append(f"  pb {pb} opn_raw engine {er} python {pr} / {s}")
out.append("判定: " + ("完全一致" if not bad else "不一致あり"))
out.append("OPN の整数 − f32 (cp、 /28、 局面の手番側。 FT も f32 の基準):")
for nm, E in (("旧 (FTI32)", EO), ("新 (FC1PREC)", EN)):
    out.append(f"  {nm}: 平均 {E.mean():+.3f} / SD {E.std():.3f} / |差| の 99% 点 {np.quantile(np.abs(E), 0.99):.2f}  "
               + " ".join(f"pb{b} 平均 {E[PB == b].mean():+.3f} SD {E[PB == b].std():.3f}" for b in (0, 1)))
print("\n".join(out))
io.open(outp, "w", encoding="utf-8").write("\n".join(out) + "\n")
