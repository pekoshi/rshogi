# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""篩: net の静的評価が「10 手先の両エンジン 1 秒探索の平均」をどれだけ当てるか (順位相関 / 手番ごとに計算して平均)。
   GPU 不要。 1 net あたり静的評価 13 万局面 (約 2 分)。 基準 3 本 (現職 top3q-102 / v3.1 / v4) との対の差に局単位 bootstrap の 2SD を付ける。
usage: py -3 screen_future.py <label> <quantised.bin>    (同じ exe で読める net のみ)
"""
import collections, io, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4evalana")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimism"))
import p0, m2_games_static as m2
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
label, net = sys.argv[1], sys.argv[2]
cache = "E:/shogi/p2shogi/tmp_mob/v4evalana/screen_%s.jsonl" % label
R = [json.loads(l) for l in io.open(m2.OUT, encoding="utf-8")]
if not os.path.exists(cache):
    p0.NET = net
    out = m2.run_static("ours", [r["sfen"] for r in R])
    with io.open(cache, "w", encoding="utf-8", newline="\n") as f:
        for x in out:
            f.write(json.dumps(dict(s=x["so"])) + "\n")
for r, l, l5 in zip(R, io.open(cache, encoding="utf-8"), io.open("E:/shogi/p2shogi/tmp_mob/v4evalana/m5_v31.jsonl", encoding="utf-8")):
    r["sx"] = json.loads(l)["s"]; r["s31"] = json.loads(l5)["s31"]
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
        if r["incheck"] or max(abs(r["so"]), abs(r["sv"]), abs(r["s31"]), abs(r["sx"])) > 1500:
            continue
        f = [term if j >= n or sc[j] is None else float(np.clip(sc[j], -CL, CL)) for j in (i + D, i + D + 1)]
        sgn = 1.0 if r["mover"] == "ours" else -1.0
        X.append((sum(f) / 2, sgn * r["so"], sgn * r["sv"], sgn * r["s31"], sgn * r["sx"], 1.0 if r["mover"] == "ours" else 0.0, r["bucket"], gi))
X = np.array(X); F, so, sv, s3, sx, mo, bk, gid = X.T
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
print("篩 %s (%s) n %d / 現職との静的の相関 %.4f / SD 比 (対 現職) %.3f" % (label, net, len(F), np.corrcoef(sx, so)[0, 1], sx.std() / so.std()))
for name, m in (("bucket 0-2", bk <= 2), ("bucket 3-7", bk >= 3)):
    ug = np.unique(gid[m]); idx = {u: np.where(m & (gid == u))[0] for u in ug}; rng = np.random.default_rng(11); d = []
    f = lambda x, Fp, mop: np.mean([sp(x[mop == v], Fp[mop == v]) for v in (0.0, 1.0)])
    base = [f(c[m], F[m], mo[m]) for c in (so, s3, sv, sx)]
    for _ in range(100):
        p = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        d.append(f(sx[p], F[p], mo[p]) - f(so[p], F[p], mo[p]))
    print("  %s | 現職 %.4f  v3.1 %.4f  v4 %.4f  %s %.4f | %s - 現職 %+.4f ±%.4f" % (name, base[0], base[1], base[2], label, base[3], label, base[3] - base[0], 2 * np.std(d)))
