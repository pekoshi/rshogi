# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""M2: 対 v4 の棋譜 3 本の全局面で、両者の静的評価を取る。 2026-11-12 (v4evalana)
  エンジンは 1 本ずつ。 探索はしない (eval / e のみ)。
  出力 E:/shogi/p2shogi/tmp_mob/v4evalana/m2_games.jsonl (1 局面 1 行)
    gid, file, ply, sfen, mover (ours/v4), sc (指した側の探索値 = 手番視点), res (手番視点 1/0.5/0), incheck,
    so (我々の静的 cp), so_raw, so_main, so_psqt, bucket, eff_fv, sv (v4 の静的 cp)
"""
import collections, io, json, os, re, subprocess, sys
import cshogi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimism"))
import p0

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SP = "C:/project/p2shogi/runs/selfplay"
FILES = [("KHV4/l1kh-vs-nag4.jsonl", "l1kh", "nag4"), ("KHPGOV4/l1khpgo-vs-nag4.jsonl", "l1khpgo", "nag4"),
         ("B40/b40-vs-nag4_minicon.jsonl", "b40", "nag4_minicon")]
OUT = "E:/shogi/p2shogi/tmp_mob/v4evalana/m2_games.jsonl"


def load():
    rows = []
    for fn, ours, opp in FILES:
        games = collections.defaultdict(list); res = {}
        for ln in io.open(os.path.join(SP, fn), encoding="utf-8"):
            o = json.loads(ln)
            if o["type"] == "move":
                games[o["game_id"]].append(o)
            elif o["type"] == "result":
                res[o["game_id"]] = o
        if not rows:
            k = next(iter(res)); print("result の例", res[k]); print("move の例", {a: b for a, b in games[k][0].items() if a != "eval"}, games[k][0].get("eval"))
        for gid, mv in games.items():
            if gid not in res:
                continue
            mv.sort(key=lambda o: o["ply"])
            rows.append((fn, gid, mv, res[gid], ours))
    return rows


def run_static(kind, sfens):
    e = p0.Eng(kind)
    print(kind, "pid", e.p.pid, "n", len(sfens))
    out = []
    try:
        B = 2000
        for i in range(0, len(sfens), B):
            chunk = sfens[i:i + B]
            for s in chunk:
                e.w("position sfen " + s)
                e.w("eval" if kind == "ours" else "e")
            for _ in chunk:
                if kind == "ours":
                    l = e.wait(lambda l: "raw_total=" in l, 120)
                    d = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", l)}
                    out.append(dict(so=d["cp"], so_raw=d["raw_total"], so_main=d["main_raw"], so_psqt=d["psqt"],
                                    bucket=d["bucket"], eff_fv=d["eff_fv"]))
                else:
                    l = e.wait(lambda l: l.startswith("eval = "), 120)
                    out.append(dict(sv=int(l.split()[2])))
            if (i // B) % 10 == 0:
                print(kind, i)
    finally:
        e.close()
    return out


def main():
    G = load()
    recs = []
    for fn, gid, mv, res, ours in G:
        w = res.get("winner") or res.get("result") or res.get("outcome")
        b = cshogi.Board()
        for o in mv:
            sf = o["sfen_before"]
            b.set_sfen(sf)
            stm = sf.split()[1]
            ev = o.get("eval") or {}
            sc = ev.get("score_cp")
            if sc is None and ev.get("score_mate") is not None:
                sc = 30000 if ev["score_mate"] > 0 else -30000
            recs.append(dict(file=fn.split("/")[0], gid=gid, ply=o["ply"], sfen=sf, stm=stm,
                             mover="ours" if o["engine"] == ours else "v4", sc=sc, nodes=ev.get("nodes"), depth=ev.get("depth"),
                             incheck=bool(b.is_check()), res_raw=res))
    print("局面", len(recs))
    if len(sys.argv) > 1 and sys.argv[1] == "dry":
        return
    sf = [r["sfen"] for r in recs]
    for kind in ("ours", "v4"):
        for r, d in zip(recs, run_static(kind, sf)):
            r.update(d)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("done", OUT)


if __name__ == "__main__":
    main()
