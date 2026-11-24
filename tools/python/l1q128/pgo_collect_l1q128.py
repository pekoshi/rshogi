# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
# KH 入り OPN 最終版の PGO 採取 (2026-11-15)。 fable2048/pgo_collect.py の写しを KH 付きに合わせた。
#   py -3 pgo_collect_l1q128.py <instr.exe> <OPN の net> <本体の net>   (fable2048r2/pgo_collect_r2.py の写し。 違いは本体の net が引数なことだけ)
# 構成 (Threads 1・定跡なし・本番と同じ LS / FV の設定、 本体 = L1 a-4 + KHFile kh_final):
#   1. none    : OPN なし (L1+KH)         序盤 g3_a/b/c 各 2 + 中盤 + 終盤 2
#   2. replace : 置き換え (L1+KH + F2, λ=256)  g3_a 4 / g3_b 4 / g3_c 2 + 中盤 1
# 🚨Fable の報告: LLVM_PROFILE_FILE が効かず cwd に default.profraw が出た。 run ごとに cwd を分けたうえで
#   LLVM_PROFILE_FILE も run ごとの絶対パス (<cwd>/<run>/env_%p.profraw) で渡し、 どちらに出たかを後で確かめる。
import io, os, sys, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
EXE = sys.argv[1]
OPN = sys.argv[2] if len(sys.argv) > 2 else "E:/shogi/p2shogi/tmp_mob/opnF2/opnF2-2/quantised.bin"
NODES = 250000
NET = sys.argv[3]   # L1Q128 (2026-11-16): 本体の net を引数で渡す (新 net で採取する)
KH = "E:/shogi/p2shogi/tmp_mob/khhr/kh_final.bin"
PROG = "E:/shogi/engines/nagisa_v31/NAGISA_V3/eval/progress.bin"
G = "C:/project/p2shogi/runs/opnint/g3/"
def poss(f, n):
    return [l.strip() for l in io.open(G + f, encoding="utf-8") if l.strip().startswith("position")][:n]
LATE = [
    "position sfen ln1g1g1nl/1ks2r3/1pppp1bpp/p4pp2/7P1/2P1P4/PPBP1PP1P/2S1GR3/LN1GK1SNL b s 25",
    "position sfen l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
    "position sfen 2+R2lgs1/3B1pg2/p2+R2+bnk/6ppg/5L3/6PPP/4g+s1K1/9/7N1 w 2SNL4Pnl7p 184",
]
# R2 (2026-11-15 Fable): ply 24 の互角局面 (OPN だけで評価する帯 = 置き換えの序盤の大半) を 4 局面足す。
import json
K24 = ["position sfen " + json.loads(l)["sfen"] for l in io.open("C:/project/p2shogi/runs/opnkpi/sets.jsonl", encoding="utf-8")][5::70][:4]
RUNS = [
    ("none", "", poss("g3_a.txt", 2) + poss("g3_b.txt", 2) + poss("g3_c.txt", 2) + LATE),
    ("replace", "OPNFile=%s" % OPN, poss("g3_a.txt", 4) + poss("g3_b.txt", 4) + poss("g3_c.txt", 2) + K24 + LATE[:1]),
]
for name, extra, ps in RUNS:
    os.makedirs(name, exist_ok=True)
    env = dict(os.environ, LLVM_PROFILE_FILE=os.path.abspath(os.path.join(name, "env_%p.profraw")))
    p = subprocess.Popen([EXE], cwd=name, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    opts = ["usi", "setoption name USI_OwnBook value false", "setoption name Threads value 1", "setoption name USI_Hash value 256",
            "setoption name PvInterval value 0", "setoption name UseMateSolverAsync value false",
            "setoption name LS_BUCKET_MODE value progress8kpabs", "setoption name LS_PROGRESS_COEFF value " + PROG,
            "setoption name FV_SCALE value 28", "setoption name FV_SCALE_OPENING value 26", "setoption name FV_SCALE_ENDGAME value 28",
            "setoption name EvalFile value " + NET, "setoption name KHFile value " + KH]
    for kv in [x for x in extra.split(";") if x]:
        k, _, v = kv.partition("="); opts.append("setoption name " + k + " value " + v)
    p.stdin.write("\n".join(opts) + "\n"); p.stdin.flush()
    tot = 0; seen = []
    for pos in ps:
        p.stdin.write("isready\n"); p.stdin.flush()
        while True:
            ln = p.stdout.readline()
            if not ln or ln.startswith("readyok"): break
            if "loaded" in ln or "Error" in ln or "No such option" in ln:
                if ln.strip() not in seen: seen.append(ln.strip())
        p.stdin.write(pos + "\ngo nodes %d\n" % NODES); p.stdin.flush(); nd = 0
        while True:
            ln = p.stdout.readline()
            if not ln: break
            if " nodes " in ln:
                tok = ln.split()
                try: nd = int(tok[tok.index("nodes") + 1])
                except Exception: pass
            if ln.startswith("bestmove"): break
        tot += nd
    p.stdin.write("quit\n"); p.stdin.flush(); rc = p.wait(timeout=120)
    print("collect %-8s 局面 %d  node 計 %d  exit %s" % (name, len(ps), tot, rc))
    for s in seen: print("    ", s[:260])
