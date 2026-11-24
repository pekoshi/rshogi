# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""篩の「対の差」: 2 本の net (どちらも screen_<label>.jsonl のキャッシュが在ること) の bucket 0-2 単独の順位相関を
   同じ局面・同じ局の bootstrap で比べる。 screen_future.py は「対 現職」の差しか出さず、 OPN どうしの差の 2SD が分からないため。
   エンジンも GPU も使わない。 2026-11-14 OPNARCH
   測り方は screen_future.py と同じ (10 手先の探索値 F / 手番ごとの順位相関の平均 / 局単位 bootstrap)。
   除外条件 (|cp| > 1500) は両方の net を含めて掛けるので、 単独の値は screen_future.py と小数 3 桁目が少し違いうる。
   2026-11-14 F1: bucket 0-1 の行を足した (OPN を bucket 0-1 だけで使う決定に合わせた判定線用。 他の行は不変)。
usage: py -3 pair_screen.py <labelA> <labelB> [bootstrap 回数=400]
"""
import collections, ctypes, io, json, sys
import numpy as np
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
D0 = "E:/shogi/p2shogi/tmp_mob/v4evalana/"
la, lb = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 400
R = [json.loads(l) for l in io.open(D0 + "m2_games.jsonl", encoding="utf-8")]
def col(path, key):
    v = [json.loads(l)[key] for l in io.open(path, encoding="utf-8")]
    if len(v) != len(R):
        raise SystemExit("🚨行数が m2 と合わない: %s (%d / %d)" % (path, len(v), len(R)))
    return v
for r, a, b, c in zip(R, col(D0 + "screen_%s.jsonl" % la, "s"), col(D0 + "screen_%s.jsonl" % lb, "s"), col(D0 + "m5_v31.jsonl", "s31")):
    r["sa"], r["sb"], r["s31"] = a, b, c
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
        if r["incheck"] or max(abs(r["so"]), abs(r["sv"]), abs(r["s31"]), abs(r["sa"]), abs(r["sb"])) > 1500:
            continue
        f = [term if j >= n or sc[j] is None else float(np.clip(sc[j], -CL, CL)) for j in (i + D, i + D + 1)]
        sgn = 1.0 if r["mover"] == "ours" else -1.0
        X.append((sum(f) / 2, sgn * r["so"], sgn * r["sa"], sgn * r["sb"], 1.0 if r["mover"] == "ours" else 0.0, r["bucket"], gi))
F, so, sa, sb, mo, bk, gid = np.array(X).T
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
f = lambda x, Fp, mop: np.mean([sp(x[mop == v], Fp[mop == v]) for v in (0.0, 1.0)])
for name, m in (("bucket 0-2", bk <= 2), ("bucket 0-1", bk <= 1), ("bucket 0", bk == 0), ("bucket 1", bk == 1), ("bucket 2", bk == 2)):
    ug = np.unique(gid[m]); idx = {u: np.where(m & (gid == u))[0] for u in ug}; rng = np.random.default_rng(11); d = []
    base = [f(c[m], F[m], mo[m]) for c in (so, sa, sb)]
    for _ in range(NB):
        p = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        d.append(f(sb[p], F[p], mo[p]) - f(sa[p], F[p], mo[p]))
    print("  %-10s n %6d | 現職 %.4f  %s %.4f  %s %.4f | %s - %s %+.4f ±%.4f (2SD / 局単位 bootstrap %d 回) / 2 本の相関 %.3f" % (
        name, m.sum(), base[0], la, base[1], lb, base[2], lb, la, base[2] - base[1], 2 * np.std(d), NB, np.corrcoef(sa[m], sb[m])[0, 1]))
