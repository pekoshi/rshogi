# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""篩の bucket 0-1 限定の「対 現職」: OPN を bucket 0-1 だけで使う決定 (11-14 ユーザー) の判定用。 元は pair_screen.py。
   各 net (どちらも screen_<label>.jsonl のキャッシュが在ること) の bucket 0-2 単独の順位相関を
   同じ局面・同じ局の bootstrap で比べる。 screen_future.py は「対 現職」の差しか出さず、 OPN どうしの差の 2SD が分からないため。
   エンジンも GPU も使わない。 2026-11-14 OPNARCH
   測り方は screen_future.py と同じ (10 手先の探索値 F / 手番ごとの順位相関の平均 / 局単位 bootstrap)。
   除外条件 (|cp| > 1500) は両方の net を含めて掛けるので、 単独の値は screen_future.py と小数 3 桁目が少し違いうる。
usage: py -3 b01_screen.py <label> [<label> ...]   (各 net を 現職 と bucket 0-1 / 0 / 1 で対比較)
"""
import collections, ctypes, io, json, sys
import numpy as np
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
D0 = "E:/shogi/p2shogi/tmp_mob/v4evalana/"
LABELS = sys.argv[1:]; NB = 400
R = [json.loads(l) for l in io.open(D0 + "m2_games.jsonl", encoding="utf-8")]
def col(path, key):
    v = [json.loads(l)[key] for l in io.open(path, encoding="utf-8")]
    if len(v) != len(R):
        raise SystemExit("🚨行数が m2 と合わない: %s (%d / %d)" % (path, len(v), len(R)))
    return v
for L in LABELS:
    for r, a in zip(R, col(D0 + "screen_%s.jsonl" % L, "s")):
        r["x_" + L] = a
KQ = 1.154; CL = 1500; D = 10
G = collections.defaultdict(list)
for r in R:
    G[(r["file"], r["gid"])].append(r)
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
f = lambda x, Fp, mop: np.mean([sp(x[mop == v], Fp[mop == v]) for v in (0.0, 1.0)])
for L in LABELS:
    X = []
    for gi, (key, mv) in enumerate(G.items()):
        mv.sort(key=lambda r: r["ply"]); n = len(mv)
        w = mv[0]["res_raw"].get("winner"); term = 0.0 if not w else (CL if "nag4" not in w else -CL)
        sc = [None if r["sc"] is None else (r["sc"] if r["mover"] == "ours" else -r["sc"] / KQ) for r in mv]
        for i, r in enumerate(mv):
            if r["incheck"] or r["bucket"] > 1 or max(abs(r["so"]), abs(r["x_" + L])) > 1500:
                continue
            fv = [term if j >= n or sc[j] is None else float(np.clip(sc[j], -CL, CL)) for j in (i + D, i + D + 1)]
            sgn = 1.0 if r["mover"] == "ours" else -1.0
            X.append((sum(fv) / 2, sgn * r["so"], sgn * r["x_" + L], 1.0 if r["mover"] == "ours" else 0.0, r["bucket"], gi))
    F, so, sx, mo, bk, gid = np.array(X).T
    for name, m in (("bucket 0-1", bk <= 1), ("bucket 0", bk == 0), ("bucket 1", bk == 1)):
        ug = np.unique(gid[m]); idx = {u: np.where(m & (gid == u))[0] for u in ug}; rng = np.random.default_rng(11); d = []
        b0, b1 = f(so[m], F[m], mo[m]), f(sx[m], F[m], mo[m])
        for _ in range(NB):
            p = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            d.append(f(sx[p], F[p], mo[p]) - f(so[p], F[p], mo[p]))
        print("%-16s %-10s n %6d | 現職 %.4f  OPN %.4f | OPN - 現職 %+.4f ±%.4f (2SD)" % (L, name, m.sum(), b0, b1, b1 - b0, 2 * np.std(d)))
