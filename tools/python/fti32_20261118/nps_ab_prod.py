# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""FTI32 S2: exe 同士の NPS 比を CPU 時間で測る (runs/speedupA/nps_cputime_ab.py の写し、 2026-11-18)。
違い: ① option は本番 (it09_capsee = 対局時の全 option、 KH / OPN 込み、 Threads 1 / Hash 256)。 net / OPN は exe ごとに渡す。
      ② 探索木が違う exe (FT の量子化が違う) も比べるので、 nodes の一致を要求しない (--gate で要求する)。 比は 1 周ごとの
         (B の nodes / B の CPU 秒) / (A の nodes / A の CPU 秒)。 局面は 16 (fti32_verify1.py と同じ)。
usage: nps_ab_prod.py <exeA> <exeB> --rounds 64 --nodes 1000000 --netA P --netB P --opnA P --opnB P [--gate]
"""
import os
import math
import queue
import re
import statistics as st
import subprocess
import sys
import threading
import time

import psutil

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "C:/project/p2shogi"
NET = "E:/shogi/bullet-shogi-nodchip-v2/arm_plane/plane-36/quantised.bin"
# 🚨2026-10-21: 旧 run が消えていて FileNotFoundError になった。
#   --common で渡せるようにし、 既定も現存する判定 run に向ける。
COMMON = ROOT + "/runs/selfplay/MIDBASEM_n200000/common.txt"
STARTPOS = ROOT + "/runs/speedupA/startpos_ply9.txt"


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EXE_A, EXE_B = sys.argv[1], sys.argv[2]
ROUNDS = int(opt("--rounds", "12"))
NODES = int(opt("--nodes", "1000000"))
NET_A = opt("--netA", NET)
NET_B = opt("--netB", NET)
COMMON = opt("--common", COMMON)

import io, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blunders_20261116"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4confirm"))
import eng, v4c  # noqa: E402
J = json.load(io.open("C:/project/p2shogi/runs/v4confirm/it09_capsee.json", encoding="utf-8"))
_o, _ = v4c.our_options(J)
options = [(k, v) for k, v in _o if k not in set(eng.DROPK) | {"EvalFile", "KHFile", "OPNFile"}] + [
    ("KHFile", "E:/shogi/p2shogi/tmp_mob/khhr/kh_x125.bin"), ("USI_OwnBook", "false"), ("Threads", "1"), ("USI_Hash", "256"),
    ("PvInterval", "0"), ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MinimumThinkingTime", "1")]
OPN_A = opt("--opnA", J["opnfile"])
OPN_B = opt("--opnB", J["opnfile"])
GATE = "--gate" in sys.argv
# 11-18 追加: 腕ごとの option の上書き (--optA K=V / --optB K=V、 繰り返し可) と局面ファイル (--posfile、 1 行 1 局面)。
EXTRA_A = [sys.argv[i + 1].split("=", 1) for i, a in enumerate(sys.argv) if a == "--optA"]
EXTRA_B = [sys.argv[i + 1].split("=", 1) for i, a in enumerate(sys.argv) if a == "--optB"]

G = "C:/project/p2shogi/runs/opnint/g3/"
sfens = [l.strip()[len("position "):] for f in ("g3_a.txt", "g3_b.txt", "g3_c.txt") for l in io.open(G + f, encoding="utf-8")
         if l.strip().startswith("position")][:12] + [
    "sfen ln1g1g1nl/1ks2r3/1pppp1bpp/p4pp2/7P1/2P1P4/PPBP1PP1P/2S1GR3/LN1GK1SNL b s 25",
    "sfen l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
    "sfen 2+R2lgs1/3B1pg2/p2+R2+bnk/6ppg/5L3/6PPP/4g+s1K1/9/7N1 w 2SNL4Pnl7p 184",
    "sfen lr4knl/3g2g2/2n1ppsp1/ppp3p1p/3pS2P1/P1P2PP1P/1P1PPsN2/2GK1G3/LN5RL b BSb 43"]


def norm_pos(l):
    l = l.strip()
    if l.startswith("position "):
        l = l[len("position "):]
    return l if l.startswith(("sfen", "startpos")) else "sfen " + l


class Engine:
    """起動したままにして 1 局面ずつ go を投げる。 CPU 時間を go の前後で差分する。"""

    def __init__(self, exe, net, opn, extra=()):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                  errors="replace", bufsize=1)
        self.ps = psutil.Process(self.p.pid)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.send("usi")
        self.wait_for("usiok")
        for k, v in options:
            self.send("setoption name %s value %s" % (k, v))
        self.send("setoption name EvalFile value " + net)
        self.send("setoption name OPNFile value " + opn)
        for k, v in extra:
            self.send("setoption name %s value %s" % (k, v))
        self.send("isready")
        self.wait_for("readyok")

    def _rd(self):
        for l in self.p.stdout:
            self.q.put(l)
        self.q.put(None)

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait_for(self, prefix, timeout=600):
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            try:
                l = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if l is None:
                raise RuntimeError("エンジンが落ちた")
            if l.startswith("info") and " nodes " in l:
                last = l
            if l.startswith(prefix):
                return last
        raise RuntimeError("timeout: " + prefix)

    def cpu(self):
        t = self.ps.cpu_times()
        return t.user + t.system

    def go(self, sfen):
        """1 局面を go nodes して (nodes, CPU 秒) を返す。"""
        self.send("usinewgame")
        self.send("isready")
        self.wait_for("readyok")
        self.send("position " + sfen)
        c0 = self.cpu()
        self.send("go nodes %d" % NODES)
        last = self.wait_for("bestmove")
        c1 = self.cpu()
        if last is None:
            raise RuntimeError("bestmove の前に info 行が無い")
        return int(re.search(r" nodes (\d+)", last).group(1)), c1 - c0

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


print("起動中 (両方を立ち上げたまま 1 局面ずつ交互に回す)")
if "--posfile" in sys.argv:
    sfens = [norm_pos(l) for l in io.open(opt("--posfile", ""), encoding="utf-8") if l.strip()]
print("  上書き A %s / B %s" % (EXTRA_A, EXTRA_B))
ea = Engine(EXE_A, NET_A, OPN_A, EXTRA_A)
eb = Engine(EXE_B, NET_B, OPN_B, EXTRA_B)
print("  A=%s" % EXE_A)
print("  B=%s" % EXE_B)
print("  局面 %d / nodes %d / rounds %d" % (len(sfens), NODES, ROUNDS))

ratios = []
pairs = []          # (round, 局面, A の CPU 秒, B の CPU 秒)
ta_tot = tb_tot = 0.0
try:
    for r in range(ROUNDS):
        ca = cb = 0.0
        nsa = nsb = 0
        for i, s in enumerate(sfens):
            b_first = ((r + i) % 2) == 1          # ABBA: 局面ごとに順序を反転
            if b_first:
                nb, tb = eb.go(s)
                na, ta = ea.go(s)
            else:
                na, ta = ea.go(s)
                nb, tb = eb.go(s)
            if GATE and na != nb:
                print("  ★round %d 局面 %d: nodes 不一致 A=%d B=%d = 探索木が違う。 比は無意味"
                      % (r + 1, i, na, nb))
                sys.exit(2)
            ca += ta
            cb += tb
            nsa += na
            nsb += nb
            pairs.append((r, i, ta, tb))
        ratios.append((nsb / cb) / (nsa / ca))
        ta_tot += ca
        tb_tot += cb
        print("  round %2d: A %8.2f s %d nodes / B %8.2f s %d nodes  B の NPS 比 %.4f"
              % (r + 1, ca, nsa, cb, nsb, ratios[-1]), flush=True)
finally:
    ea.close()
    eb.close()

lg = [math.log(x) for x in ratios]
gm = math.exp(st.fmean(lg))
if len(lg) > 1:
    se = st.stdev(lg) / math.sqrt(len(lg))
    lo, hi = math.exp(st.fmean(lg) - 1.96 * se), math.exp(st.fmean(lg) + 1.96 * se)
else:
    lo = hi = gm
print("")
print("B の NPS 比 ((B の nodes / CPU 秒) / (A の nodes / CPU 秒))")
print("  幾何平均 %.4f (%+.2f%%)   95%% 区間 [%+.2f%%, %+.2f%%]   2σ 幅 ±%.2f%%"
      % (gm, (gm - 1) * 100, (lo - 1) * 100, (hi - 1) * 100, (hi - lo) / 2 * 100))
print("  中央値   %.4f (%+.2f%%)  最小 %.4f 最大 %.4f"
      % (st.median(ratios), (st.median(ratios) - 1) * 100, min(ratios), max(ratios)))
print("  合計 CPU  A %.1f s / B %.1f s" % (ta_tot, tb_tot))

# ---- 局面ごとの対で集計する (A と B が最も近接しているのは同じ局面の直後どうし)
#   🚨同じ周の対は独立でないので、 **周でクラスタしたブートストラップ**で区間を出す。
if pairs:
    import random
    by_round = {}
    for r_, i_, ta_, tb_ in pairs:
        by_round.setdefault(r_, []).append(math.log(ta_ / tb_))
    keys = list(by_round)
    allr = [x for k in keys for x in by_round[k]]
    pm = math.exp(st.fmean(allr))
    rb = random.Random(29)
    bs = []
    for _ in range(4000):
        pick = [by_round[keys[rb.randrange(len(keys))]] for _ in keys]
        flat = [x for g in pick for x in g]
        bs.append(math.exp(st.fmean(flat)))
    bs.sort()
    blo, bhi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]
    print("")
    print("  局面ごとの対で集計 (n=%d 対 / %d 周でクラスタ)" % (len(pairs), len(keys)))
    print("    幾何平均 %.4f (%+.2f%%)   95%% 区間 [%+.2f%%, %+.2f%%]   2σ 幅 ±%.2f%%"
          % (pm, (pm - 1) * 100, (blo - 1) * 100, (bhi - 1) * 100, (bhi - blo) / 2 * 100))
print("  ★区間が 0%% を含むなら『速くなった』とは言えない")
