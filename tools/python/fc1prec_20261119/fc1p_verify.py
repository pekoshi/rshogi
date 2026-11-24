# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""FC1PREC 段 3 (a) 本体: python の整数模擬とエンジンの静的評価の完全一致 (runs/fti32_20261118/fti32_verify.py の写し)。

  py -3 fc1p_verify.py <exe (P2_DIAG + P2_FC1_PREC)> <新形式 net> <weights.bin> <m2_games.jsonl> <out.txt> [N=1000] [seed=20261118]

違い: fc 層 (fc0 / fc1 / fc2 の bias と重み) は**新形式 net のファイルから**読む (書き出し器の式を python で作り直さない = 書き出しの誤りも捕まる)。
FT は従来どおり weights.bin から k=8 の式 (fti32_write.py と同じ。 FT は今回変えていない)。 後段は P2_FC1_PREC の式:
  l2in = [clip(floor(v²/2^19), 0, 508), clip(floor(v/2^5), 0, 508)]   (L1Q128: 旧は 2^21 / 2^7、 0..127)
  rl   = clip(floor((B1 + W1·l2in) / 2^7), 0, 508)
  main = floor((B2 + W2·rl + 2) / 4) + l1[skip]
エンジンは P2_NOTHREAT=1 (threat / KH を足さない = python と同じ式) で起動する。
"""
import io
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np
import cshogi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k256_20261117"))
from bullet_inference_v19 import unpack_yo_hcp, encode_halfka  # noqa: E402
from k256_merge import read_weight  # noqa: E402

NB, L1, M, L3, FT, HALF, IN, L2IN = 8, 16, 15, 64, 1024, 512, 73305, 30
K, MM = 8, 32
exe, net, wpath, jl, outp = sys.argv[1:6]
N = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 20261118

T = {n: (read_weight(wpath, n) if n == "l0w" else read_weight(wpath, n).astype(np.float64)) for n in ("l0w", "l0b", "psqtw")}
W = np.round(np.float32(254.0 * K) * T["l0w"].reshape(IN, FT))
B = np.round(254.0 * K * T["l0b"])
psq = np.round(8128.0 * T["psqtw"].reshape(IN, NB))
del T
# fc 層をファイルの末尾 8 bucket から読む
fb = open(net, "rb").read()
ver = struct.unpack_from("<I", fb, 0)[0]
assert ver == 0x7AF32F22, hex(ver)
bs = 4 + 4 * L1 + L1 * FT + 4 * L3 + L3 * 32 + 4 + L3
st = len(fb) - NB * bs
B0, W0, B1, W1, B2, W2 = [], [], [], [], [], []
for k in range(NB):
    p = st + k * bs + 4
    B0.append(np.frombuffer(fb, "<i4", L1, p).astype(np.float64)); p += 4 * L1
    W0.append(np.frombuffer(fb, "i1", L1 * FT, p).reshape(L1, FT).astype(np.float64)); p += L1 * FT
    B1.append(np.frombuffer(fb, "<i4", L3, p).astype(np.float64)); p += 4 * L3
    W1.append(np.frombuffer(fb, "i1", L3 * 32, p).reshape(L3, 32)[:, :L2IN].astype(np.float64)); p += L3 * 32
    B2.append(float(struct.unpack_from("<i", fb, p)[0])); p += 4
    W2.append(np.frombuffer(fb, "i1", L3, p).astype(np.float64))
del fb


def py_eval(sfen, b):
    board = cshogi.Board(sfen)
    buf = np.zeros(32, np.uint8)
    board.to_psfen(buf)
    fs, fn = encode_halfka(unpack_yo_hcp(bytes(buf)))
    fs, fn = np.asarray(fs), np.asarray(fn)
    As = np.clip(B + W[fs].sum(0, dtype=np.float64), 0, 254 * K)
    An = np.clip(B + W[fn].sum(0, dtype=np.float64), 0, 254 * K)
    dv = 512.0 * K * K / MM
    TR = np.r_[np.floor(As[:HALF] * As[HALF:] / dv), np.floor(An[:HALF] * An[HALF:] / dv)]
    l1 = np.floor((MM * B0[b] + W0[b] @ TR) / MM)
    v = l1[:M]
    l2in = np.r_[np.clip(np.floor(v * v / 2 ** 19), 0, 508), np.clip(np.floor(v / 32), 0, 508)]
    rl = np.clip(np.floor((B1[b] + W1[b] @ l2in) / 128), 0, 508)
    main = np.floor((B2[b] + rl @ W2[b] + 2) / 4) + l1[M]
    D = psq[fs, b].sum() - psq[fn, b].sum()
    ps = int(np.sign(D) * np.floor(abs(D) / 2))
    return int(main), ps


R = [json.loads(l) for l in io.open(jl, encoding="utf-8")]
rng = np.random.default_rng(SEED)
idx = rng.choice([i for i, r in enumerate(R) if not r.get("incheck")], size=N, replace=False)
env = dict(os.environ, P2_NOTHREAT="1")
p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     encoding="utf-8", errors="replace", env=env, bufsize=1)


def send(s):
    p.stdin.write(s + "\n"); p.stdin.flush()


def until(pref):  # pref で始まる行か、 その文字列を含む行まで読む
    lines = []
    while True:
        l = p.stdout.readline()
        if not l:
            raise SystemExit("エンジンが落ちた:\n" + "".join(lines[-20:]))
        lines.append(l)
        if l.startswith(pref) or (pref.startswith('info string ') and pref in l):
            return lines


send("usi"); until("usiok")
for k_, v_ in (("Threads", "1"), ("USI_Hash", "16"), ("EvalFile", net), ("USI_OwnBook", "false"),
               ("LS_PROGRESS_COEFF", "E:/shogi/engines/nagisa_v31/NAGISA_V3/eval/progress.bin"), ("LS_BUCKET_MODE", "progress8kpabs"),
               ("KHFile", "E:/shogi/p2shogi/tmp_mob/khhr/kh_x125.bin"),
               ("OPNFile", "E:/shogi/p2shogi/tmp_mob/opnF2Qx/opnF2Qx-16/quantised_fti32k8_fc1p.bin")):
    send(f"setoption name {k_} value {v_}")
send("isready")
head = until("readyok")
info = [l.strip() for l in head if "P2NNUE" in l and ("FT" in l or "arch" in l or "FC1PREC" in l)]
bad = []
n = 0
for i in idx:
    sfen = R[i]["sfen"]
    send("position sfen " + sfen)
    send("eval")
    ls = until("info string P2NNUE bucket=")   # isready は net を読み直すので使わない (memory: gate_isready_was_the_bottleneck)
    m = [re.search(r"P2NNUE bucket=(\d+) main_raw=(-?\d+) psqt=(-?\d+)", l) for l in ls]
    m = [x for x in m if x]
    if len(m) != 1:
        raise SystemExit("eval の行が取れない:\n" + "".join(ls[-10:]))
    b, er, ep = int(m[0].group(1)), int(m[0].group(2)), int(m[0].group(3))
    pr, pp = py_eval(sfen, b)
    n += 1
    if (pr, pp) != (er, ep):
        bad.append((sfen, b, er, pr, ep, pp))
send("quit")
p.wait(timeout=60)
out = [f"FC1PREC 検証 (a) 本体 exe {exe}", f"net {net} / 局面 {n} / 不一致 {len(bad)}"] + info
for s, b, er, pr, ep, pp in bad[:5]:
    out.append(f"  bucket {b} main_raw engine {er} python {pr} / psqt engine {ep} python {pp} / {s}")
out.append("判定: " + ("完全一致" if not bad else "不一致あり"))
print("\n".join(out))
io.open(outp, "w", encoding="utf-8").write("\n".join(out) + "\n")
