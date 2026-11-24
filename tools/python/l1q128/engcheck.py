# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""L1Q128 の照合 (DESIGN-20261116-L1Q128-FABLE.md 3.5 / 検証 a・b-①)。 2026-11-16
   runs/mainq/mainq_dump.py の写しから P2_MAINQ_OUT を外したもの。 本番 define の exe の `eval` が出す cp= を、
   篩の 130,965 局面 (m2_games.jsonl) で screen_<ラベル>.jsonl と比べて不一致の行数を出す。
   設定は screen_L1khon を作った器と同じ p0.ours_opts + KHFile。 OPN は指定しない (= 本体だけの値)。 エンジン 1 本・IDLE。

usage: py -3 engcheck.py <exe> <net> <ラベル> <64|128> [開始行 終了行]
   例: py -3 engcheck.py <新 exe> .../quantised_l1q128.bin L1q128_Q1 128     (検証 a)
       py -3 engcheck.py <新 exe> .../quantised.bin       L1khon    64      (検証 b-①)
   第 4 引数 = 読み込み行 "P2NNUE L1 main scale: <n>" に期待する値 (違えば照合せずに止める)。
"""
import os
import ctypes, io, json, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4evalana")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimism"))
import p0  # noqa: E402
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
KHF = "E:/shogi/p2shogi/tmp_mob/khhr/kh_final.bin"
D0 = "E:/shogi/p2shogi/tmp_mob/v4evalana/"
if len(sys.argv) not in (5, 7):
    raise SystemExit(__doc__)
exe, net, label, want = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
a, z = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) == 7 else (0, None)
R = [json.loads(l)["sfen"] for l in io.open(D0 + "m2_games.jsonl", encoding="utf-8")][a:z]
ref = [json.loads(l)["s"] for l in io.open(D0 + "screen_%s.jsonl" % label, encoding="utf-8")][a:z]
assert len(R) == len(ref), (len(R), len(ref))
t0 = time.time()
p0.NET = net
e = p0.Eng("ours", exe=exe, extra=[("KHFile", KHF)])
print("exe %s\nnet %s\nラベル %s / pid %d / n %d / missing opts %s" % (exe, net, label, e.p.pid, len(R), e.missing))
for s in e.loaded:
    print("  " + s)
# 読み込み行の目盛りを確かめる (isready をもう 1 回送って読み込みの行を拾う)
scale = []
e.w("isready")
e.wait(lambda l: l.startswith("readyok"), 300,
       collect=lambda l: scale.append(int(re.search(r"L1 main scale: (\d+)", l).group(1))) if "L1 main scale:" in l else None)
print("  L1 main scale (読み込み行): %s" % scale)
if scale != [want]:
    e.close()
    raise SystemExit("🚨目盛りが期待 (%d) と違う: %s。 照合しない" % (want, scale))
cps = []
try:
    B = 2000
    for i in range(0, len(R), B):
        chunk = R[i:i + B]
        for s in chunk:
            e.w("position sfen " + s); e.w("eval")
        for _ in chunk:
            l = e.wait(lambda l: "raw_total=" in l, 120)
            cps.append(int(re.search(r" cp=(-?\d+)", l).group(1)))
        if (i // B) % 10 == 0:
            print("  %d / %d  %.0fs" % (a + i, a + len(R), time.time() - t0))
finally:
    e.close()
bad = [k for k, (x, y) in enumerate(zip(cps, ref)) if x != y]
print("結果: %s / %d 局面 / screen_%s と cp 不一致 %d 行 %s (%.0fs)" % (
    label, len(cps), label, len(bad), ("先頭 " + str([(a + k, cps[k], ref[k]) for k in bad[:10]])) if bad else "", time.time() - t0))
