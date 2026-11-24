# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""MAINQ 3: 篩の「対の差」を bucket 0 / 1 / 2 / 3 / 4-7 / 0-1 / 0-2 / 3-7 で出す。 2026-11-15
   測り方は runs/opnarch/pair_screen.py と同じ (10 手先の両エンジン探索の平均との手番ごとの順位相関の平均 / 局単位 bootstrap)。
   除外 (|cp| > 1500) は 基準・比べる全 label を含めて 1 回だけ掛けるので、 全ての対が同じ局面集合で比べられる。
usage: py -3 mainq_screen.py <基準 label> <label> [<label> ...] [-n 400]
   出力: 各帯で 基準 と各 label の値、 label - 基準 と 2SD (局単位 bootstrap)。 基準を "L1khon" にすると整数 (現職 L1 の本物) と比べる。
"""
import collections, ctypes, io, json, sys
import numpy as np
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
D0 = "E:/shogi/p2shogi/tmp_mob/v4evalana/"
args = sys.argv[1:]
NBOOT = 400
if "-n" in args:
    i = args.index("-n"); NBOOT = int(args[i + 1]); del args[i:i + 2]
base, labs = args[0], args[1:]
R = [json.loads(l) for l in io.open(D0 + "m2_games.jsonl", encoding="utf-8")]


def col(path, key):
    v = [json.loads(l)[key] for l in io.open(path, encoding="utf-8")]
    if len(v) != len(R):
        raise SystemExit("🚨行数が m2 と合わない: %s (%d / %d)" % (path, len(v), len(R)))
    return v


ALL = [base] + labs
V = {L: col(D0 + "screen_%s.jsonl" % L, "s") for L in ALL}
s31 = col(D0 + "m5_v31.jsonl", "s31")
for i, r in enumerate(R):
    r["s31"] = s31[i]; r["L"] = [V[L][i] for L in ALL]
KQ = 1.154; CL = 1500; D = 10
G = collections.defaultdict(list)
for r in R:
    G[(r["file"], r["gid"])].append(r)
X = []
for gi, (key, mv) in enumerate(G.items()):
    mv.sort(key=lambda r: r["ply"]); n = len(mv)
    w = mv[0]["res_raw"].get("winner"); term = 0.0 if not w else (CL if "nag4" not in w else -CL)
    sc = [None if r["sc"] is None else (r["sc"] if r["mover"] == "ours" else -r["sc"] / KQ) for r in mv]
    for i, r in enumerate(mv):
        if r["incheck"] or max([abs(r["so"]), abs(r["sv"]), abs(r["s31"])] + [abs(x) for x in r["L"]]) > 1500:
            continue
        f = [term if j >= n or sc[j] is None else float(np.clip(sc[j], -CL, CL)) for j in (i + D, i + D + 1)]
        sgn = 1.0 if r["mover"] == "ours" else -1.0
        X.append([sum(f) / 2, 1.0 if r["mover"] == "ours" else 0.0, r["bucket"], gi] + [sgn * x for x in r["L"]])
X = np.array(X); F, mo, bk, gid = X[:, :4].T; S = X[:, 4:]
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
f2 = lambda x, Fp, mop: np.mean([sp(x[mop == v], Fp[mop == v]) for v in (0.0, 1.0)])
BANDS = (("bucket 0", bk == 0), ("bucket 1", bk == 1), ("bucket 2", bk == 2), ("bucket 3", bk == 3), ("bucket 4-7", bk >= 4),
         ("bucket 0-1", bk <= 1), ("bucket 0-2", bk <= 2), ("bucket 2-7", bk >= 2), ("bucket 3-7", bk >= 3))
print("篩 基準 %s / 比べる %s / n %d / bootstrap %d 回" % (base, " ".join(labs), len(F), NBOOT))
for name, m in BANDS:
    ug = np.unique(gid[m]); idx = {u: np.where(m & (gid == u))[0] for u in ug}; rng = np.random.default_rng(11)
    val = [f2(S[m, k], F[m], mo[m]) for k in range(len(ALL))]
    d = [[] for _ in labs]
    for _ in range(NBOOT):
        p = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        b0 = f2(S[p, 0], F[p], mo[p])
        for k in range(len(labs)):
            d[k].append(f2(S[p, k + 1], F[p], mo[p]) - b0)
    print("  %-10s n %6d | %s %.4f | " % (name, m.sum(), base, val[0]) + " | ".join(
        "%s %.4f (%+.4f ±%.4f)" % (L, val[k + 1], val[k + 1] - val[0], 2 * np.std(d[k])) for k, L in enumerate(labs)))
