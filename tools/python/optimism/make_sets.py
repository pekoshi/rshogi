# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""OPTIMISM 前段の局面セット PRE / NEG / POS を作る。 2026-11-12。
設計: runs/prereg/DESIGN-20261112-OPTIMISM.md (判定線はそちらに固定)

入力 (対 v4 の時間制 1 手 1 秒の棋譜 / jsonl):
  KHV4/l1kh-vs-nag4.jsonl          我々 = l1kh     相手 = nag4
  KHPGOV4/l1khpgo-vs-nag4.jsonl    我々 = l1khpgo  相手 = nag4
  B40/b40-vs-nag4_minicon.jsonl    我々 = b40      相手 = nag4_minicon
  🚨game_id はファイルごとの連番 -> 局の鍵は "<ファイル名>#<game_id>"
  🚨eval.score_cp は**手番視点** (指した側の視点)。 相手の値を我々視点にするときは符号反転
  score_mate は ±30000 に写す (符号は mate の符号)

定義 (局面はすべて**我々の手番**の sfen_before):
  PRE  負け局。 転機 T = 相手 (v4) の値を我々視点にしたものが -300 を割り、 以後最後まで戻らない最初の手 (ply)。
       窓 A = [T-20, T-11] / 窓 B = [T-10, T-2] から我々手番の局面を取る。
       局の優先: KH 系 2 ファイルの負け局を先 (予備集計と同じ母集団) -> 足りない分を B40。
       1 局 = 窓 A 1 + 窓 B 1 (= 最大 2、 上限 3 を守る)。 目標 300 (A 150 / B 150)。
  NEG  勝ち・引き分け局。 我々の値 |s| <= 150、 かつ以後 20 手 (ply) の
       全ての「我々の値 (ply q) と直後の v4 の値 (ply q+1、 我々視点)」の対が ±60 以内。
       PRE の (ply 帯 20 刻み x 持駒層) の分布に合わせて層別に 150 (最善努力 / 不足は近い層から)。 1 局最大 3。
  POS  負け局。 我々の値 s0 (-300 <= s0 <= 600 = 転落前) から、 以後 6 手 (ply p+2/p+4/p+6 の我々の値) の最小が
       s0 - 150 以下。 1 局 1 局面で 60。 PRE と同一 sfen は除く。
  a20 部分集合 (P0 の 2,000 万ノード): PRE 120 (A 60 + B 60、 すべて別の局) / NEG 60 / POS 60。

position 文字列: "sfen <ply 1 の sfen_before> moves m1 .. m_{p-1}" (千日手検出のため手順付き)。
  手順の復元は cshogi で ply ごとの sfen_before と突き合わせて検算する (不一致の局は捨てて件数を印字)。

出力: runs/optimism/sets.jsonl (1 行 1 局面)、 runs/optimism/sets_summary.txt
"""
import collections
import io
import json
import os
import random
import sys

import cshogi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SP = "C:/project/p2shogi/runs/selfplay"
OUT = "C:/project/p2shogi/runs/optimism"
FILES = [("KHV4/l1kh-vs-nag4.jsonl", "l1kh", "nag4", 0),
         ("KHPGOV4/l1khpgo-vs-nag4.jsonl", "l1khpgo", "nag4", 0),
         ("B40/b40-vs-nag4_minicon.jsonl", "b40", "nag4_minicon", 1)]
MATE = 30000
SEED = 20261112


def val(ev):
    if not ev:
        return None
    if ev.get("score_cp") is not None:
        return float(ev["score_cp"])
    if ev.get("score_mate") is not None:
        m = ev["score_mate"]
        return float(MATE if m > 0 else -MATE) if m != 0 else None
    return None


def hand_count(sfen):
    h = sfen.split()[2]
    if h == "-":
        return 0
    n, num = 0, ""
    for ch in h:
        if ch.isdigit():
            num += ch
        else:
            n += int(num) if num else 1
            num = ""
    return n


def hand_layer(n):
    return 0 if n <= 2 else 1 if n <= 5 else 2 if n <= 9 else 3


def ply_band(p):
    return min(p // 20, 9)


def load(fn, ours, opp, prio):
    games = collections.defaultdict(list)
    res = {}
    for ln in io.open(os.path.join(SP, fn), encoding="utf-8"):
        o = json.loads(ln)
        if o["type"] == "move":
            games[o["game_id"]].append(o)
        elif o["type"] == "result":
            res[o["game_id"]] = o
    out = []
    bad = 0
    for gid, mv in games.items():
        if gid not in res:
            continue
        mv.sort(key=lambda o: o["ply"])
        if [o["ply"] for o in mv] != list(range(1, len(mv) + 1)):
            bad += 1
            continue
        b = cshogi.Board()
        b.set_sfen(mv[0]["sfen_before"])
        ok = True
        for o in mv:
            if b.sfen().rsplit(" ", 1)[0] != o["sfen_before"].rsplit(" ", 1)[0]:
                ok = False
                break
            b.push_usi(o["move_usi"])
        if not ok:
            bad += 1
            continue
        w = res[gid].get("winner")
        outcome = "win" if w == ours else ("loss" if w == opp else "draw")
        # our-view value per ply
        seq = []
        for o in mv:
            v = val(o.get("eval"))
            mine = o["engine"] == ours
            seq.append(dict(ply=o["ply"], mine=mine, raw=v,
                            ov=(v if mine else (-v if v is not None else None)),
                            sfen=o["sfen_before"], move=o["move_usi"]))
        out.append(dict(key="%s#%d" % (os.path.basename(fn), gid), src=fn.split("/")[0], prio=prio,
                        outcome=outcome, reason=res[gid].get("reason"), start=mv[0]["sfen_before"],
                        moves=[o["move_usi"] for o in mv], seq=seq))
    print("  %s: 局 %d (手順不一致で捨てた %d)" % (fn, len(out), bad))
    return out


def turning_point(g):
    """v4 の値 (我々視点) が -300 を割って以後戻らない最初の ply。 無ければ None。"""
    opp = [x for x in g["seq"] if not x["mine"] and x["ov"] is not None]
    if not opp or opp[-1]["ov"] >= -300:
        return None
    t = None
    for x in reversed(opp):
        if x["ov"] < -300:
            t = x["ply"]
        else:
            break
    return t


def rec(g, x, kind, **kw):
    p = x["ply"]
    pos = "sfen %s moves %s" % (g["start"], " ".join(g["moves"][:p - 1])) if p > 1 else "sfen " + g["start"]
    hc = hand_count(x["sfen"])
    d = dict(set=kind, game=g["key"], src=g["src"], outcome=g["outcome"], ply=p, sfen=x["sfen"],
             position=pos.strip(), played=x["move"], s_game=x["ov"], hand=hc, hl=hand_layer(hc),
             pb=ply_band(p))
    d.update(kw)
    return d


def ours_at(g, p):
    for x in g["seq"]:
        if x["ply"] == p:
            return x
    return None


def main():
    rng = random.Random(SEED)
    G = []
    for fn, ours, opp, prio in FILES:
        G += load(fn, ours, opp, prio)
    loss = [g for g in G if g["outcome"] == "loss"]
    wd = [g for g in G if g["outcome"] != "loss"]
    print("  負け %d / 勝ち・分 %d" % (len(loss), len(wd)))

    # ---------- PRE
    cand = []
    noT = 0
    for g in loss:
        T = turning_point(g)
        if T is None:
            noT += 1
            continue
        A = [x for x in g["seq"] if x["mine"] and x["ov"] is not None and abs(x["ov"]) < MATE
             and T - 20 <= x["ply"] <= T - 11]
        B = [x for x in g["seq"] if x["mine"] and x["ov"] is not None and abs(x["ov"]) < MATE
             and T - 10 <= x["ply"] <= T - 2]
        if A and B:
            cand.append((g, T, A, B))
    print("  PRE 候補局 %d (転機なし %d)" % (len(cand), noT))
    rng.shuffle(cand)
    cand.sort(key=lambda c: c[0]["prio"])          # KH 系を先に (安定ソート = 中はシャッフル順)
    pre = []
    pre_games = cand[:150]
    for i, (g, T, A, B) in enumerate(pre_games):
        xa, xb = rng.choice(A), rng.choice(B)
        # a20 部分集合: 局 0-59 の窓 A / 局 60-119 の窓 B (= 120 局面すべて別の局)
        pre.append(rec(g, xa, "PRE", T=T, win="A", dT=xa["ply"] - T, a20=(i < 60)))
        pre.append(rec(g, xb, "PRE", T=T, win="B", dT=xb["ply"] - T, a20=(60 <= i < 120)))
    pre_sfens = set(r["sfen"] for r in pre)

    # ---------- NEG (PRE の ply 帯 x 持駒層に合わせる)
    strata = collections.Counter((r["pb"], r["hl"]) for r in pre)
    ncand = collections.defaultdict(list)       # stratum -> [(g, x)]
    for g in wd:
        byp = dict((x["ply"], x) for x in g["seq"])
        el = []
        for x in g["seq"]:
            if not x["mine"] or x["ov"] is None or abs(x["ov"]) > 150:
                continue
            p = x["ply"]
            if p + 20 > len(g["seq"]):
                continue
            ok = True
            for q in range(p, p + 20, 2):      # 我々 q と v4 q+1 の対 (10 対)
                a, b = byp.get(q), byp.get(q + 1)
                if a is None or b is None or a["ov"] is None or b["ov"] is None or abs(a["ov"] - b["ov"]) > 60:
                    ok = False
                    break
            if ok:
                el.append(x)
        for x in el:
            ncand[(ply_band(x["ply"]), hand_layer(hand_count(x["sfen"])))].append((g, x))
    # 層ごとの割当 = 150 x PRE の比率 (最大剰余)。 1 局 1 局面
    tot = sum(strata.values())
    quota = dict((k, 150.0 * v / tot) for k, v in strata.items())
    alloc = dict((k, int(q)) for k, q in quota.items())
    for k in sorted(quota, key=lambda k: -(quota[k] - int(quota[k])))[:150 - sum(alloc.values())]:
        alloc[k] += 1
    # 🚨「静か + v4 と 20 手一致」は序盤に偏る (ply 60 以上は 12 局面しか無い) -> 厳密な層合わせは不可能。
    #   次善: 1 局最大 3 局面 (同局内は 6 ply 以上離す)、 層の不足は最も近い層 (|ply 帯差| + |持駒層差|) から補う。
    per = collections.defaultdict(list)
    neg = []

    def take(g, x, k_target, fill):
        ps = per[g["key"]]
        if len(ps) >= 3 or any(abs(p - x["ply"]) < 6 for p in ps):
            return False
        ps.append(x["ply"])
        neg.append(rec(g, x, "NEG", fill=fill, tgt=list(k_target)))
        return True
    allc = [(k, g, x) for k, l in ncand.items() for g, x in l]
    rng.shuffle(allc)
    # 1 局 1 局面を優先するため、 まだ使っていない局を先に試す 2 周
    short = {}
    for k in sorted(alloc, key=lambda k: len(ncand.get(k, []))):
        got = 0
        for rnd in (0, 1):
            for kk, g, x in allc:
                if got >= alloc[k] or kk != k:
                    continue
                if rnd == 0 and per[g["key"]]:
                    continue
                if take(g, x, k, False):
                    got += 1
        short[k] = alloc[k] - got
    for k, s in sorted(short.items()):
        if s <= 0:
            continue
        near = sorted(allc, key=lambda c: (abs(c[0][0] - k[0]) + abs(c[0][1] - k[1]), -c[0][0], len(per[c[1]["key"]])))
        for kk, g, x in near:
            if s <= 0:
                break
            if take(g, x, k, True):
                s -= 1
    print("  NEG 候補 %d 局面 / 採用 %d (層不足で近い層から補充 %d)" % (sum(len(v) for v in ncand.values()), len(neg),
                                                    sum(1 for r in neg if r.get("fill"))))
    rng.shuffle(neg)
    for i, r in enumerate(neg):
        r["a20"] = i < 60

    # ---------- POS
    pcand = []
    for g in loss:
        byp = dict((x["ply"], x) for x in g["seq"])
        el = []
        for x in g["seq"]:
            if not x["mine"] or x["ov"] is None or not (-300 <= x["ov"] <= 600) or x["sfen"] in pre_sfens:
                continue
            fut = [byp.get(x["ply"] + d) for d in (2, 4, 6)]
            fut = [f["ov"] for f in fut if f is not None and f["ov"] is not None]
            if len(fut) == 3 and min(fut) <= x["ov"] - 150:
                el.append((x, min(fut) - x["ov"]))
        if el:
            pcand.append((g, el))
    rng.shuffle(pcand)
    pcand.sort(key=lambda c: c[0]["prio"])
    pos = []
    for g, el in pcand[:60]:
        x, drop = rng.choice(el)
        pos.append(rec(g, x, "POS", drop6=drop, a20=True))
    print("  POS 候補局 %d / 採用 %d" % (len(pcand), len(pos)))

    allr = pre + neg + pos
    for i, r in enumerate(allr):
        r["id"] = "%s%03d" % (r["set"], i)
    with io.open(os.path.join(OUT, "sets.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # summary
    L = []
    for nm, rs in (("PRE", pre), ("NEG", neg), ("POS", pos)):
        gs = set(r["game"] for r in rs)
        src = collections.Counter(r["src"] for r in rs)
        L.append("%s: %d 局面 / %d 局 / a20 %d / 出典 %s" % (nm, len(rs), len(gs), sum(r["a20"] for r in rs), dict(src)))
        L.append("   ply 帯 %s" % sorted(collections.Counter(r["pb"] * 20 for r in rs).items()))
        L.append("   持駒層 %s" % sorted(collections.Counter(r["hl"] for r in rs).items()))
        L.append("   我々の値 (棋譜) 中央 %.0f" % sorted(r["s_game"] for r in rs)[len(rs) // 2])
    L.append("PRE 窓: %s" % dict(collections.Counter(r["win"] for r in pre)))
    txt = "\n".join(L)
    print(txt)
    io.open(os.path.join(OUT, "sets_summary.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
