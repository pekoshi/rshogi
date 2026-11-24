# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""NPS の A/B を **CPU 時間**で測る。 2026-09-29 (再作成)。

## なぜ壁時計が使えないか

このマシンは floodgate 1 本 + リーグ 2 本 + 学習を同時に抱えているので、
壁時計の NPS は **2σ ±3.74%** (log 比の SD 7.6%) も振れる。 1〜2% の税を
測るには桁が足りない。

✅ **プロセスの CPU 時間 (user + system) を使う**。 他プロセスに CPU を取られても、
   自分が回した分だけが増えるので、 負荷変動にほぼ不感。 実測分解能 **±0.69%**。

## 3 つの仕掛け

1. **CPU 時間** — `psutil.Process.cpu_times()` の user + system。 子プロセスは無し。
2. **ABBA** — 🚨単純な ABAB だと**順序効果 +2.45%** が乗る (09-29 に実測)。
   1 周ごとに ABBA / BAAB を入れ替えて相殺する。
3. **局面ごとの対応差** — 局面で NPS が 2 倍違うので、 同じ局面の A と B を対にする。
   比の対数を取って平均 (= 幾何平均)、 **周でクラスタした bootstrap** で区間を出す。

## 使い方

    py -3 runs/rescore/nps_cputime_ab.py <A.exe> <B.exe> [周回数] [ノード数]

判定は「B は A に対して 何 % か」。 税 = 100 - その値。

🚨ノード数が A/B で一致しない周は捨てる (一致しないなら探索木が違う =
  NPS の比較として無効。 ★19 の「eval bit-exact だけでは不十分」の裏返し)。
"""
import io
import math
import random
import statistics
import subprocess
import sys
import time

import psutil

# 🚨line_buffering が無いと 8 KB たまるまで flush されず途中経過が見えない (09-28 に踏んだ)。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

NET = "E:/shogi/bullet-shogi-nodchip-v2/_kept_nets/ctl-61.bin"
# 第 5 引数で net を差し替えられる (案1 G2 では p=0 の P2TPRJ 付き net を使う)。
net_override = None
PROG = "E:/shogi/engines/nagisa_v31/NAGISA_V3/eval/progress.bin"

# 序盤 / 中盤 / 終盤入口 / 終盤を混ぜる (税は局面で変わるので 1 局面で測らない)
POSS = [
    "position startpos moves 7g7f 3c3d 2g2f 4c4d",
    "position sfen ln1g1g1nl/1ks2r3/1pppp1bpp/p4pp2/7P1/2P1P4/PPBP1PP1P/2S1GR3/LN1GK1SNL b s 25",
    "position sfen l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
    "position sfen 2+R2lgs1/3B1pg2/p2+R2+bnk/6ppg/5L3/6PPP/4g+s1K1/9/7N1 w 2SNL4Pnl7p 184",
]


def bench_one(exe, pos, nodes, net=None, which="A"):
    """1 局面を go nodes して (CPU 秒, ノード数) を返す。 CPU 秒は user+system。"""
    p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    ps = psutil.Process(p.pid)
    # 🚨option 名は regime_nps.py 準拠。 存在しない名前 (P2_PROGRESS_FILE 等) は
    #   黙って無視され、 progress が読まれないまま測ることになる。
    # 🚨Threads=1 を必ず入れる。 既定のままだと並列探索でノード数が非決定的になり、
    #   ノード一致の検査で対が全部捨てられる。
    opts = [
        "usi",
        "setoption name USI_OwnBook value false",
        "setoption name Threads value 1",
        "setoption name USI_Hash value 256",
        "setoption name PvInterval value 0",
        "setoption name UseMateSolverAsync value false",
        "setoption name LS_BUCKET_MODE value progress8kpabs",
        "setoption name LS_PROGRESS_COEFF value " + PROG,
        "setoption name FV_SCALE value 28",
        "setoption name FV_SCALE_OPENING value 26",
        "setoption name FV_SCALE_ENDGAME value 28",
        "setoption name EvalFile value " + (net or net_override or NET),
        "isready",
    ]
    # 2026-11-08: 腕ごとの USI option (--opt-a / --opt-b)。 同じ exe で option だけ
    #   変える A/B (例 P2QSearchThreatScale=0) を測るため。 セミコロン区切り。
    _flag = "--opt-a" if which == "A" else "--opt-b"
    extra = sys.argv[sys.argv.index(_flag) + 1] if _flag in sys.argv else ""
    for kv in [x.strip() for x in extra.split(";") if x.strip()]:
        k, _, v = kv.partition("=")
        opts.append("setoption name " + k.strip() + " value " + v.strip())
    send = "".join(o + chr(10) for o in opts)
    p.stdin.write(send)
    p.stdin.flush()
    # readyok まで待つ (ここまでの CPU は計らない)
    while True:
        ln = p.stdout.readline()
        if not ln or ln.startswith("readyok"):
            break
    try:
        c = ps.cpu_times()
        t0 = c.user + c.system
    except Exception:
        t0 = 0.0
    p.stdin.write("usinewgame\n" + pos + "\ngo nodes %d\n" % nodes)
    p.stdin.flush()
    nd = 0
    while True:
        ln = p.stdout.readline()
        if not ln:
            break
        if " nodes " in ln:
            tok = ln.split()
            try:
                nd = int(tok[tok.index("nodes") + 1])
            except Exception:
                pass
        if ln.startswith("bestmove"):
            break
    try:
        c = ps.cpu_times()
        t1 = c.user + c.system
    except Exception:
        t1 = t0
    try:
        p.stdin.write("quit\n")
        p.stdin.flush()
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return t1 - t0, nd


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    exe_a, exe_b = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000
    # 🚨別 net を比べるときは --net-a / --net-b を使う。 探索木が違うのでノード数は
    #   一致しない -> **CPU / ノード**で正規化して比べる (ノード一致の要求は外す)。
    net_a = net_b = None
    av = sys.argv
    if "--net-a" in av:
        net_a = av[av.index("--net-a") + 1]
    if "--net-b" in av:
        net_b = av[av.index("--net-b") + 1]
    per_node = bool(net_a or net_b)
    if not per_node and len(av) > 5 and not av[5].startswith("--"):
        globals()["net_override"] = av[5]
        print("  net = " + av[5])
    if per_node:
        print("  net A = %s" % (net_a or net_override or NET))
        print("  net B = %s" % (net_b or net_override or NET))
        print("  🚨別 net なので **CPU / ノード**で比べる (ノード数の一致は要求しない)")

    print("A = %s" % exe_a)
    print("B = %s" % exe_b)
    print("%d 周 x %d 局面 x %d ノード、 ABBA、 CPU 時間" % (rounds, len(POSS), nodes))
    print("")

    # ratios[i] = 各対の log(cpu_B / cpu_A)。 round_id も持って bootstrap をクラスタ化する。
    logs, rids, dropped = [], [], 0
    for r in range(rounds):
        for pi, pos in enumerate(POSS):
            # 🚨ABBA: 周ごとに順序を入れ替えて順序効果 (+2.45%) を相殺する。
            order = ["A", "B", "B", "A"] if (r % 2 == 0) else ["B", "A", "A", "B"]
            ta, tb, na, nb = [], [], [], []
            for who in order:
                exe = exe_a if who == "A" else exe_b
                t, nd = bench_one(exe, pos, nodes, net_a if who == "A" else net_b, who)
                (ta if who == "A" else tb).append(t)
                (na if who == "A" else nb).append(nd)
            if per_node:
                if not na or not nb or sum(na) <= 0 or sum(nb) <= 0:
                    dropped += 1
                    continue
                ca = statistics.mean(t / n for t, n in zip(ta, na))
                cb = statistics.mean(t / n for t, n in zip(tb, nb))
                if ca <= 0 or cb <= 0:
                    dropped += 1
                    continue
                logs.append(math.log(cb / ca))
                rids.append(r)
                continue
            if not na or not nb or min(na) != max(na) or min(nb) != max(nb) or na[0] != nb[0]:
                # ノード数が違う = 探索木が違う。 NPS 比として無効なので捨てる。
                dropped += 1
                continue
            ca, cb = statistics.mean(ta), statistics.mean(tb)
            if ca <= 0 or cb <= 0:
                dropped += 1
                continue
            logs.append(math.log(cb / ca))
            rids.append(r)
        if logs:
            cur = math.exp(statistics.mean(logs))
            print("  周 %2d 終了  ここまで B/A の CPU 比 %.4f  (B の NPS = A の %.2f%%)"
                  % (r + 1, cur, 100.0 / cur))

    if not logs:
        print("★★有効な対が 0。 ノード数が一致しなかった (捨てた %d)" % dropped)
        return

    # 周でクラスタした bootstrap (同じ周の対は同じ負荷を共有するので独立でない)
    rng = random.Random(20260929)
    by_round = {}
    for lg, r in zip(logs, rids):
        by_round.setdefault(r, []).append(lg)
    keys = list(by_round)
    boots = []
    for _ in range(4000):
        s = []
        for _ in range(len(keys)):
            s.extend(by_round[rng.choice(keys)])
        boots.append(math.exp(statistics.mean(s)))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    ratio = math.exp(statistics.mean(logs))

    print("")
    print("=== 結果  有効な対 %d / 捨てた %d" % (len(logs), dropped))
    print("  B/A の CPU 時間比   %.4f   95%% [%.4f, %.4f]" % (ratio, lo, hi))
    print("  B の NPS = A の     %.2f%%   95%% [%.2f%%, %.2f%%]"
          % (100.0 / ratio, 100.0 / hi, 100.0 / lo))
    tax = 100.0 * (ratio - 1.0)
    print("  → 税 %+.2f%%   95%% [%+.2f%%, %+.2f%%]"
          % (tax, 100.0 * (lo - 1.0), 100.0 * (hi - 1.0)))


if __name__ == "__main__":
    main()
