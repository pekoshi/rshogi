# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""lzA 段 1 の bit-exact 関門 (2026-11-16 / 設計 runs/prereg/DESIGN-20261116-THREATINC-FABLE.md §4)。
   G-a = runs/pgo_positions.txt 49 局面 x 200K / G-b = startpos_yo2025_ply24_1000.txt 先頭 150 x 200K /
   G-c = 同 先頭 24 x 2M。 3 本とも「手が違う 0 / nodes だけ違う 0」が合格。
   option = v4c.our_options(it04_infrep.json) (本番 toml + L1Q128 net + KHFile + OPN) から時間・Threads・Hash 系を
   eng.py と同じく差し替えたもの。 P2InferiorRepHistory は false / true の両方で base と new を比べる。
   局面ごとに isready (TT/history を消す)。 エンジンは同時に 1 本だけ。 base の自己再現 (base2, off, G-a のみ) も取る。
usage: py -3 -u gate.py <base.exe> <new.exe>"""
import os
import io, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blunders_20261116"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4confirm"))
import eng, v4c  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
BASE, NEW = sys.argv[1], sys.argv[2]
IT04 = "C:/project/p2shogi/runs/v4confirm/it04_infrep.json"
OUT = "C:/project/p2shogi/runs/lzA_20261116/"
R = "C:/project/p2shogi/runs/"


def opts():
    o, _ = v4c.our_options(json.load(io.open(IT04, encoding="utf-8")))
    o = [(k, v) for k, v in o if k not in eng.DROPK and k != "P2InferiorRepHistory"]
    return o + [("USI_OwnBook", "false"), ("Threads", "1"), ("USI_Hash", "256"), ("PvInterval", "0"),
                ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MinimumThinkingTime", "1")]


def mk(exe, infrep):
    eng.load = lambda kind, hash_mb=512: (exe, opts())
    e = eng.Eng("ours", extra=[("P2InferiorRepHistory", infrep)])
    if e.missing:
        e.close(); raise SystemExit("option が無い: %s" % e.missing)
    return e


def lines(f, n):
    out = []
    for x in io.open(R + f, encoding="utf-8").read().splitlines():
        x = x.strip()
        if not x:
            continue
        if x.startswith("position "):
            x = x[len("position "):]
        if not (x.startswith("sfen ") or x.startswith("startpos")):
            x = "sfen " + x
        out.append(x)
    return out[:n]


GATES = [("G-a", lines("pgo_positions.txt", 49), 200000),
         ("G-b", lines("startpos_yo2025_ply24_1000.txt", 150), 200000),
         ("G-c", lines("startpos_yo2025_ply24_1000.txt", 24), 2000000)]


def run(exe, infrep, P, nodes):
    e = mk(exe, infrep)
    try:
        return [(r["bestmove"], r.get("nodes"), r.get("depth"), r.get("v"), " ".join(r.get("pv", [])))
                for r in (e.search(p, "nodes %d" % nodes) for p in P)]
    finally:
        e.close()


def cmp(tag, a, b):
    mv = sum(1 for x, y in zip(a, b) if x[0] != y[0])
    nd = sum(1 for x, y in zip(a, b) if x[0] == y[0] and x[1] != y[1])
    full = sum(1 for x, y in zip(a, b) if x == y)
    ok = (mv == 0 and nd == 0 and len(a) == len(b))
    print("  %-28s 局面 %d: 手が違う %d / nodes だけ違う %d / 全一致 (手・nodes・depth・score・pv) %d -> %s"
          % (tag, len(a), mv, nd, full, "完全一致" if ok else "食い違いあり"))
    for i, (x, y) in enumerate(zip(a, b)):
        if x[:2] != y[:2]:
            print("    [%d] base %s %s / new %s %s" % (i, x[0], x[1], y[0], y[1]))
    return ok


def main():
    print("base %s\nnew  %s\noption の元 %s" % (BASE, NEW, IT04))
    raw, allok = {}, True
    for g, P, nodes in GATES:
        for inf in ("false", "true"):
            b = run(BASE, inf, P, nodes)
            n = run(NEW, inf, P, nodes)
            raw["%s_%s" % (g, inf)] = dict(base=b, new=n)
            allok &= cmp("%s x %d infrep=%s" % (g, nodes, inf), b, n)
            if g == "G-a" and inf == "false":
                b2 = run(BASE, inf, P, nodes)
                raw["G-a_false_base2"] = b2
                cmp("(器の再現) base 対 base2", b, b2)
    json.dump(raw, io.open(OUT + "gate_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("総合: %s" % ("3 関門 x infrep off/on すべて完全一致" if allok else "🚨食い違いあり"))


main()
