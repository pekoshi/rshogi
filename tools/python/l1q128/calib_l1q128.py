# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""L1Q128 の OPN 目盛りの取り直し (DESIGN-20261116-L1Q128-FABLE.md 5 章 e-1)。 2026-11-16
   v4c.py calib (= calib_cov.py の 1/SD 比 (bucket 0-2) x 1024) と同じ選別・同じ式で、 本体の値 so を
   m2_games.jsonl の so (v4c.py calib が使う値) / screen_L1khon (現職 = 旧 net のエンジン値) / screen_<新ラベル> に
   差し替えて 3 通り出し、 新 / 旧 が 3 値とも ±1% 以内かを判定する。 エンジンは起動しない。 何も書かない。
usage: py -3 calib_l1q128.py <OPN の篩ラベル> <新 net の篩ラベル> <現行 scale 3 値 (カンマ区切り)>
   例: py -3 calib_l1q128.py opnF2Qx_sb16 L1q128_Q1 1196,1204,1377
"""
import io, json, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
V4E = "E:/shogi/p2shogi/tmp_mob/v4evalana/"
opn_lbl, new_lbl, cur = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]


def col(lbl):
    return [json.loads(l)["s"] for l in io.open(V4E + "screen_%s.jsonl" % lbl, encoding="utf-8")]


R = [json.loads(l) for l in io.open(V4E + "m2_games.jsonl", encoding="utf-8")]
sx = col(opn_lbl)
SO = {"m2.so": [r["so"] for r in R], "L1khon": col("L1khon"), new_lbl: col(new_lbl)}
assert all(len(v) == len(R) for v in list(SO.values()) + [sx])
res = {}
for name, so_all in SO.items():
    so, xx, bk = [], [], []
    for r, s, x in zip(R, so_all, sx):
        if r["incheck"] or max(abs(s), abs(r["sv"]), abs(x)) > 1500:
            continue
        sgn = 1.0 if r["mover"] == "ours" else -1.0
        so.append(sgn * s); xx.append(sgn * x); bk.append(r["bucket"])
    so, xx, bk = map(np.array, (so, xx, bk))
    res[name] = [so[bk == b].std() / xx[bk == b].std() for b in (0, 1, 2)]
    print("so = %-10s 1/SD 比 b0-2 %s -> Scale %s" % (name, ["%.5f" % k for k in res[name]], [int(round(k * 1024)) for k in res[name]]))
print("現行 scale %s" % cur)
for base in ("m2.so", "L1khon"):
    rel = [n / o - 1.0 for n, o in zip(res[new_lbl], res[base])]
    print("新 (%s) / 旧 (%s) - 1 = %s" % (new_lbl, base, ["%+.3f%%" % (100 * r) for r in rel]))
rel = [n / o - 1.0 for n, o in zip(res[new_lbl], res["L1khon"])]
new_sc = [int(round(k * 1024)) for k in res[new_lbl]]
print("判定 (旧 = L1khon との比、 ±1%%): %s / 新 net の Scale = %s / 現行 scale との比 %s" % (
    "据え置き (3 値とも ±1% 以内)" if all(abs(r) <= 0.01 for r in rel) else "**外れ = 新しい値を入れる**", new_sc,
    ["%+.3f%%" % (100 * (n / c - 1)) for n, c in zip(new_sc, cur)]))
