# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""FTI32 S1 検証 (2): python の整数模擬 (ftdiag_s0_fti32.py の (A k) と同じ式) とエンジンの静的評価の完全一致。

  py -3 fti32_verify.py <exe (P2_DIAG ビルド)> <net.bin> <weights.bin> <k> <m2_games.jsonl> <out.txt> [N=1000] [seed=20261118]

エンジンは環境変数 P2_NOTHREAT=1 (threat / KH を足さない = python と同じ式) で起動し、 KHFile / OPN は渡さない。
局面ごとに `position sfen ...` → `eval` の "info string P2NNUE bucket=.. main_raw=.. psqt=.." を読み、 python が同じ bucket で
計算した main_raw と psqt を比べる。 1 件でも違えば最初の 5 件を出す。
"""
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import cshogi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k256_20261117"))
from bullet_inference_v19 import unpack_yo_hcp, encode_halfka  # noqa: E402
from k256_merge import read_weight  # noqa: E402

NB, L1, M, L3, FT, HALF, IN = 8, 16, 15, 64, 1024, 512, 73305
exe, net, wpath, KM, jl, outp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
K, MM = (int(x) for x in (KM.split(",") + ["1"])[:2])   # "k" か "k,m" (m = FT 出力 i16 の倍率、 PREREG の (B))
N = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 20261118

T = {n: (read_weight(wpath, n) if n == "l0w" else read_weight(wpath, n).astype(np.float64)) for n in
     ("l0w", "l0b", "psqtw", "l1w", "l1fw", "l1b", "l1fb", "l2w", "l2b", "l3w", "l3b")}
# FT: fti32_write.py / ftdiag_s0_fti32.py と同じ式 (k = 1 は C7c の ftw_i と同じ)
W = np.round(np.float32(254.0 * K) * T["l0w"].reshape(IN, FT))
B = np.round(254.0 * K * T["l0b"])
psq = np.round(8128.0 * T["psqtw"].reshape(IN, NB))
merged = T["l1w"].reshape(FT, NB, L1) + T["l1fw"].reshape(FT, L1)[:, None, :]
biasf = T["l1b"].reshape(NB, L1) + T["l1fb"][None, :]
SC = np.r_[np.full(M, 128.0), 64.0]; BS = np.r_[np.full(M, 16256.0), 8128.0]   # L1Q128
W0 = [np.clip(np.round(SC[:, None] * merged[:, b, :].T), -128, 127) for b in range(NB)]
B0 = [np.round(BS * biasf[b]) for b in range(NB)]
l2wf = T["l2w"].reshape(2 * M, NB, L3)
W1 = [np.clip(np.round(64.0 * l2wf[:, b, :].T), -128, 127) for b in range(NB)]
B1 = [np.round(8128.0 * T["l2b"][b * L3:(b + 1) * L3]) for b in range(NB)]
W2 = [np.clip(np.round(64.0 * T["l3w"].reshape(L3, NB)[:, b]), -128, 127) for b in range(NB)]
B2 = [np.round(8128.0 * T["l3b"][b]) for b in range(NB)]
del T


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
    l2in = np.r_[np.clip(np.floor(v * v / 2 ** 21), 0, 127), np.clip(np.floor(v / 128), 0, 127)]
    rl = np.clip(np.floor((B1[b] + W1[b] @ l2in) / 64), 0, 127)
    main = B2[b] + rl @ W2[b] + l1[M]
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
               ("KHFile", "E:/shogi/p2shogi/tmp_mob/khhr/kh_x125.bin"), ("OPNFile", "E:/shogi/p2shogi/tmp_mob/opnF2Qx/opnF2Qx-16/quantised.bin")):
    send(f"setoption name {k_} value {v_}")
send("isready")
head = until("readyok")
info = [l.strip() for l in head if "P2NNUE" in l and ("FT" in l or "arch" in l)]
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
out = [f"FTI32 検証 (2) exe {exe}", f"net {net} / k={K} m={MM} / 局面 {n} / 不一致 {len(bad)}"] + info
for s, b, er, pr, ep, pp in bad[:5]:
    out.append(f"  bucket {b} main_raw engine {er} python {pr} / psqt engine {ep} python {pp} / {s}")
out.append("判定: " + ("完全一致" if not bad else "不一致あり"))
print("\n".join(out))
io.open(outp, "w", encoding="utf-8").write("\n".join(out) + "\n")
