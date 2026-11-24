# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""PGO の採取 (11-23): 確定版の全 option で 100 万ノード、実戦の局面 60 個 (実手数 24-59: 15 / 60-99: 20 / 100-139: 15 / 140-: 10)。
usage: py -3 collect.py <計装 exe> <profraw の出力先>"""
import glob
import io
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4confirm"))
import v4c  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
CFG = "C:/project/p2shogi/runs/v4confirm/final_s29g_backs_sw500.json"
RUNS = ["POLOFFq_s29_v4_n300k_j100", "I29_n300k_T1_j51to100", "CMP1S_S29_j50"]
BANDS = [(24, 59, 15), (60, 99, 20), (100, 139, 15), (140, 9999, 10)]
NODES = 1000000
DROP = {"Threads", "USI_Hash", "NetworkDelay", "NetworkDelay2", "MinimumThinkingTime", "PvInterval"}


def options():
    J = json.load(io.open(CFG, encoding="utf-8"))
    v4c.KHFILE = dict(tuple(x) for x in J["extra"])["KHFile"]
    o = [(k, v) for k, v in v4c.our_options(J)[0] if k not in DROP]
    return o + [("Threads", "1"), ("USI_Hash", "512"), ("PvInterval", "0"), ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MinimumThinkingTime", "1")]


def positions():
    P = {b: set() for b in BANDS}
    for r in RUNS:
        for f in glob.glob("C:/project/p2shogi/runs/selfplay/%s/ours-vs-*.jsonl" % r):
            for l in io.open(f, encoding="utf-8"):
                if '"type":"move"' not in l:
                    continue
                d = json.loads(l)
                if d["engine"] != "ours":
                    continue
                n = int(d["sfen_before"].split()[-1])
                for b in BANDS:
                    if b[0] <= n <= b[1]:
                        P[b].add(d["sfen_before"])
    rng = random.Random(20261123)
    out = []
    for b in BANDS:
        x = sorted(P[b])
        if len(x) < b[2]:
            raise SystemExit("局面が足りない: %s" % (b,))
        out += rng.sample(x, b[2])
    return out


def main():
    exe, raw = sys.argv[1], sys.argv[2]
    os.makedirs(raw, exist_ok=True)
    env = dict(os.environ, LLVM_PROFILE_FILE=os.path.abspath(os.path.join(raw, "p_%p.profraw")))
    p = subprocess.Popen([exe], cwd=raw, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)

    def wait(tok):
        while True:
            l = p.stdout.readline()
            if not l:
                raise SystemExit("engine が終了した (%s)" % tok)
            if "No such option" in l:
                raise SystemExit(l.strip())
            if l.startswith(tok):
                return l

    p.stdin.write("usi\n"); p.stdin.flush(); wait("usiok")
    p.stdin.write("\n".join("setoption name %s value %s" % kv for kv in options()) + "\n"); p.stdin.flush()
    tot = 0
    P = positions()
    for s in P:
        p.stdin.write("isready\n"); p.stdin.flush(); wait("readyok")
        p.stdin.write("position sfen %s\ngo nodes %d\n" % (s, NODES)); p.stdin.flush()
        nd = 0
        while True:
            l = p.stdout.readline()
            if not l:
                raise SystemExit("engine が終了した (go)")
            if " nodes " in l:
                t = l.split()
                try:
                    nd = int(t[t.index("nodes") + 1])
                except (ValueError, IndexError):
                    pass
            if l.startswith("bestmove"):
                break
        tot += nd
    p.stdin.write("quit\n"); p.stdin.flush()
    rc = p.wait(timeout=120)
    print("採取 局面 %d / node 計 %d / exit %s" % (len(P), tot, rc))
    if rc != 0 or tot < len(P) * NODES * 0.9 or not glob.glob(os.path.join(raw, "*.profraw")):
        raise SystemExit("採取に異常")


main()
