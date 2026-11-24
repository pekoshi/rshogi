# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""案 2 の前段 (DESIGN-20261113-KH-FABLE.md): 同じ KH 入り PGO exe・同じ khhr/a-4 の net で、KHFile あり / なしの篩。
screen_future.py と同じ局面・同じ目標 (10 手先の両エンジン 1 秒探索の平均)。順位相関は手番ごとに計算して平均する。
角換わり (その局の最初の局面で盤上に角・馬が無い) x bucket 0-2 / 3-7 の 4 区分で、on - off の対の差を出す。
線 (事前固定): bucket 0-2 の差が ±0.003 以内なら「序盤帯では何もしていない」。
usage: py -3 screen_kh_onoff.py
"""
import collections, io, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4evalana")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimism"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import p0, m2_games_static as m2

EXE = "E:/shogi/p2shogi_cplus/release/pgo/khpgo_1112/tsumugi_khpgo_1112_pgo_static.exe"
NET = "E:/shogi/p2shogi/tmp_mob/khhr/a-4/quantised.bin"
KHF = "E:/shogi/p2shogi/tmp_mob/khhr/kh_final.bin"
R = [json.loads(l) for l in io.open(m2.OUT, encoding="utf-8")]
_Eng = p0.Eng


def get(label, extra):
    cache = "E:/shogi/p2shogi/tmp_mob/khreview/screen_kh_%s.jsonl" % label
    if not os.path.exists(cache):
        p0.NET = NET
        p0.Eng = lambda kind, exe=None, extra0=(): _Eng(kind, exe=EXE, extra=list(extra0) + extra)
        out = m2.run_static("ours", [r["sfen"] for r in R])
        p0.Eng = _Eng
        with io.open(cache, "w", encoding="utf-8", newline="\n") as f:
            for x in out:
                f.write(json.dumps(dict(s=x["so"])) + "\n")
    return [json.loads(l)["s"] for l in io.open(cache, encoding="utf-8")]


on = get("on", [("KHFile", KHF)])
off = get("off", [])
same = sum(1 for a, b in zip(on, off) if a == b)
print("on と off の静的評価が同じ局面 %d / %d (全部同じなら KHFile が効いていない)" % (same, len(on)))
for r, a, b in zip(R, on, off):
    r["on"], r["off"] = a, b
KQ = 1.154; CL = 1500; D = 10
G = collections.defaultdict(list)
for r in R:
    G[(r["file"], r["gid"])].append(r)
X = []
for gi, (key, mv) in enumerate(G.items()):
    mv.sort(key=lambda r: r["ply"]); n = len(mv)
    board0 = mv[0]["sfen"].split()[0]
    kaku = 1.0 if ("B" not in board0.upper()) else 0.0
    w = mv[0]["res_raw"].get("winner"); term = 0.0 if not w else (CL if "nag4" not in w else -CL)
    sc = [None if r["sc"] is None else (r["sc"] if r["mover"] == "ours" else -r["sc"] / KQ) for r in mv]
    for i, r in enumerate(mv):
        if r["incheck"] or max(abs(r["on"]), abs(r["off"])) > 1500:
            continue
        f = [term if j >= n or sc[j] is None else float(np.clip(sc[j], -CL, CL)) for j in (i + D, i + D + 1)]
        sgn = 1.0 if r["mover"] == "ours" else -1.0
        X.append((sum(f) / 2, sgn * r["on"], sgn * r["off"], 1.0 if r["mover"] == "ours" else 0.0, r["bucket"], gi, kaku))
X = np.array(X); F, son, soff, mo, bk, gid, kk = X.T
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
f = lambda x, Fp, mop: np.mean([sp(x[mop == v], Fp[mop == v]) for v in (0.0, 1.0)])
print("篩 KHFile あり / なし (exe khpgo_1112 / net khhr a-4) n %d / 局 %d (角換わり %d)" % (len(F), len(np.unique(gid)), len(np.unique(gid[kk == 1]))))
for name, m in (("bucket 0-2 全体", bk <= 2), ("bucket 3-7 全体", bk >= 3),
                ("bucket 0-2 角換わり", (bk <= 2) & (kk == 1)), ("bucket 0-2 それ以外", (bk <= 2) & (kk == 0)),
                ("bucket 3-7 角換わり", (bk >= 3) & (kk == 1)), ("bucket 3-7 それ以外", (bk >= 3) & (kk == 0))):
    ug = np.unique(gid[m]); idx = {u: np.where(m & (gid == u))[0] for u in ug}; rng = np.random.default_rng(11); d = []
    for _ in range(200):
        p = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        d.append(f(son[p], F[p], mo[p]) - f(soff[p], F[p], mo[p]))
    a, b = f(son[m], F[m], mo[m]), f(soff[m], F[m], mo[m])
    print("  %-18s n %6d | on %.4f  off %.4f | on - off %+.4f ±%.4f" % (name, m.sum(), a, b, a - b, 2 * np.std(d)))
print("線 (事前固定): bucket 0-2 の差が ±0.003 以内 -> 序盤帯では何もしていない")
