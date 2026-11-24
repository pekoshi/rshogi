# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""ponkotsu の score を DLSuisho の目盛りへ写す換算表 (LUT) を作り、 PSV に適用する。 2026-11-12。

ユーザー指示「目盛りは合わせて」。 学習データの主力教師 (DLSuisho) と同じ目盛りにしてから学習する。
- build: ponko2g.psv と dlsui_src.psv (同じ局面・同じ並び) から、 ponko の値 p ごとに
  E[σ(d/600) | p] を求め、 600·logit で score に戻す。 符号対称 (|p| で集めて ± に展開)、 |p| について単調に整える。
  fit_scale.py の結果: 定数倍なら a = 0.82 (σ 誤差最小) だが帯ごとに比が違う -> 表で写す。
- apply: PSV の score (offset 32-33, int16, 手番視点) を表で書き換える (その場で / チャンク単位)。
  🚨2 回適用しないよう、 適用済みの印 `<psv>.calib` を作り、 印があれば何もしない。

usage: py -3 calib_lut.py build [--n 8000000]          -> runs/ponko_novel/calib_lut.npy
       py -3 calib_lut.py apply <psv> [<psv> ...]
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
REC = 40
P = "E:/shogi/p2shogi/data/ponko2g.psv"
Dl = "E:/shogi/p2shogi/data/dlsui_src.psv"
LUT = "C:/project/p2shogi/runs/ponko_novel/calib_lut.npy"
PMAX = 4144
BIN = 16


def build(n):
    tot = 2147483600 // REC
    nb = 80
    per = n // nb
    ps, ds = [], []
    with open(P, "rb") as fp, open(Dl, "rb") as fd:
        for b in range(nb):
            off = (tot - per) * b // (nb - 1)
            fp.seek(off * REC); fd.seek(off * REC)
            a = np.frombuffer(fp.read(per * REC), dtype=np.uint8).reshape(-1, REC)
            c = np.frombuffer(fd.read(per * REC), dtype=np.uint8).reshape(-1, REC)
            same = (a[:, :32] == c[:, :32]).all(axis=1)
            ps.append(a[same, 32:34].copy().view("<i2").reshape(-1).astype(np.int32))
            ds.append(c[same, 32:34].copy().view("<i2").reshape(-1).astype(np.float64))
    p = np.concatenate(ps); d = np.concatenate(ds)
    s = np.sign(p); s[s == 0] = 1
    ap = np.abs(p)
    sd = 1 / (1 + np.exp(-(d * s) / 600.0))      # |p| 側の向きに揃えた σ
    nbins = PMAX // BIN + 1
    idx = np.minimum(ap // BIN, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins).astype(np.float64)
    sm = np.bincount(idx, weights=sd, minlength=nbins)
    mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    # 空の bin は線形補間、 単調 (非減少) に整える、 σ は (0.5, 1) に収める
    xs = np.arange(nbins)
    ok = ~np.isnan(mean)
    mean = np.interp(xs, xs[ok], mean[ok])
    mean[0] = 0.5 if cnt[0] == 0 else mean[0]
    mean = np.maximum.accumulate(np.clip(mean, 0.5, 1 - 1e-6))
    sc_bin = 600.0 * np.log(mean / (1 - mean))
    lut = np.zeros(2 * PMAX + 1, dtype=np.int16)    # 添字 = p + PMAX
    for v in range(-PMAX, PMAX + 1):
        b = min(abs(v) // BIN, nbins - 1)
        y = sc_bin[b] if v >= 0 else -sc_bin[b]
        lut[v + PMAX] = int(np.clip(round(y), -31000, 31000))
    lut[PMAX] = 0
    np.save(LUT, lut)
    print("LUT を書いた %s (標本 %d)" % (LUT, len(p)))
    for v in (0, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 4000, 4144):
        print("  p %5d -> %6d" % (v, lut[v + PMAX]))
    # 検算: 写した後の σ 誤差 (定数倍 0.82 と比較)
    q = lut[np.clip(p, -PMAX, PMAX) + PMAX].astype(np.float64)
    sdd = 1 / (1 + np.exp(-d / 600.0))
    m = (np.abs(d) < 10000)
    e_lut = np.mean((1 / (1 + np.exp(-q[m] / 600)) - sdd[m]) ** 2)
    e_082 = np.mean((1 / (1 + np.exp(-0.82 * p[m] / 600)) - sdd[m]) ** 2)
    e_1 = np.mean((1 / (1 + np.exp(-p[m] / 600)) - sdd[m]) ** 2)
    print("σ 二乗誤差: 素 %.5f / 定数倍 0.82 %.5f / 表 %.5f" % (e_1, e_082, e_lut))


def apply(paths):
    lut = np.load(LUT)
    for path in paths:
        mark = path + ".calib"
        if os.path.exists(mark):
            print("skip (適用済み) %s" % path); continue
        size = os.path.getsize(path)
        if size % REC:
            print("🚨大きさが 40 の倍数でない %s -> 飛ばす" % path); continue
        chunk = 1_600_000 * REC
        n = 0; clip = 0
        with open(path, "r+b") as f:
            pos = 0
            while pos < size:
                f.seek(pos)
                buf = bytearray(f.read(min(chunk, size - pos)))
                a = np.frombuffer(buf, dtype=np.uint8).reshape(-1, REC)
                sc = a[:, 32:34].copy().view("<i2").reshape(-1).astype(np.int32)
                clip += int((np.abs(sc) > PMAX).sum())
                new = lut[np.clip(sc, -PMAX, PMAX) + PMAX].astype("<i2")
                a[:, 32:34] = new.view(np.uint8).reshape(-1, 2)
                f.seek(pos); f.write(buf)
                pos += len(buf); n += len(a)
        with open(mark, "w", encoding="utf-8") as g:
            json.dump({"lut": LUT, "records": n, "clipped_input": clip}, g)
        print("適用 %s : %d 局面 (|p| > %d を %d 件 端に寄せた)" % (path, n, PMAX, clip))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8_000_000)
    elif len(sys.argv) >= 3 and sys.argv[1] == "apply":
        apply(sys.argv[2:])
    else:
        print(__doc__)
