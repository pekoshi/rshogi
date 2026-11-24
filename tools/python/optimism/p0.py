# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""P0 (目盛りの切り分けと「深さで見えるか」)。 2026-11-12。
設計: runs/prereg/DESIGN-20261112-OPTIMISM.md の P0 (判定線は固定 / 変えない)

  本物の楽観 = s1 - b/k (k = 回帰値 と 1.19 / 1.3 / 1.43)
  X: (s1 - a20) >= 0.40 x (s1 - b/k) -> 深い自分の探索で見える。 ②③ に実弾
  Y: 0.15〜0.40 -> ① と ②  /  Z: < 0.15 -> ②③④ を閉じ ① は G1 次第
  器の検算: POS で s1 - a20 >= 100cp / NEG で |s1 - a20| <= 30cp

測るもの (局面はすべて我々の手番 / 値は手番視点 = 我々視点):
  s1   現職 39 万ノード x 3 回 (中央値)   a2  現職 200 万   a20 現職 2,000 万 (a20 部分集合だけ)
  b    v4 200 万
器の流儀 (runs/rescore/exe_agree_posset.py を流用 + 記憶の罠):
  - Threads 1 / USI_Hash 512 / 本番 option (spsa_adopt_opts.py の出力から Book 系を除く) / EvalFile = top3q-102
  - position は "sfen <開始局面> moves ..." (千日手検出のため手順付き)
  - **測定ごとに isready** (usinewgame は no-op。 isready で TT/history を消す。 net 再ロードで約 1.75 s かかるが探索の器では外さない)
  - go nodes N -> bestmove を読んでから次へ (quit は最後だけ)
  - 結果は 1 測定 1 行で runs/optimism/p0_raw.jsonl に追記 (再起動しても済んだ測定は飛ばす)

usage:
  py -3 p0.py run [--workers 2] [--only a20sub]      測定 (中断しても再開できる)
  py -3 p0.py analyze                                 集計と判定 (runs/prereg/VERDICT-OPTIMISM-P0-20261112.log)
  py -3 p0.py test                                    1 局面で両エンジンを試す
"""
import collections
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
OUT = "C:/project/p2shogi/runs/optimism"
RAW = OUT + "/p0_raw.jsonl"
OURS = "E:/shogi/p2shogi_cplus/release/pgo/lz_1111/tsumugi_lz_1111_pgo_static.exe"
NET = "E:/shogi/p2shogi/tmp_mob/top3q/top3q-102/quantised.bin"
V4 = "E:/shogi/engines/nagisa_v4_20261101/NAGISA_V4-Windows-AVX2/NAGISA_V4.exe"
V4EVAL = "E:/shogi/engines/nagisa_v4/NAGISA_V4-Windows-AVX2/eval"
MATE = 30000
COND = {"s1": ("ours", 390000), "a2": ("ours", 2000000), "a20": ("ours", 20000000), "b": ("v4", 2000000)}
DROPK = ("BookFile", "BookDir", "BookOnTheFly", "USI_OwnBook", "EvalFile", "EvalDir", "Threads", "USI_Hash",
         "NetworkDelay", "NetworkDelay2", "MinimumThinkingTime", "MaxMovesToDraw", "USI_Ponder")


def ours_opts():
    t = io.open(OUT + "/prod_opts.txt", encoding="utf-8").read().split()
    o = []
    for kv in t:
        k, _, v = kv.partition("=")
        if any(k.startswith(d) for d in DROPK):
            continue
        o.append((k, v))
    return o + [("USI_OwnBook", "false"), ("Threads", "1"), ("USI_Hash", "512"), ("PvInterval", "0"),
                ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MinimumThinkingTime", "1"),
                ("EvalFile", NET)]


def v4_opts():
    return [("USI_OwnBook", "false"), ("Threads", "1"), ("USI_Hash", "512"), ("EvalDir", V4EVAL),
            ("FV_SCALE", "28"), ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MinimumThinkingTime", "1"),
            ("PvInterval", "0")]


class Eng:
    def __init__(self, kind, exe=None, extra=()):
        self.kind = kind
        exe = exe or (OURS if kind == "ours" else V4)
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                  bufsize=1, cwd=os.path.dirname(exe), creationflags=0x4000)  # BELOW_NORMAL
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.w("usi")
        names = set()
        self.wait(lambda l: l.startswith("usiok"), 60, collect=lambda l: names.add(l.split()[2]) if l.startswith("option name") else None)
        opts = (ours_opts() if kind == "ours" else v4_opts()) + list(extra)
        self.missing = [k for k, _ in opts if k not in names]
        for k, v in opts:
            if k in names:
                self.w("setoption name %s value %s" % (k, v))
        self.loaded = []
        self.isready(first=True)

    def _rd(self):
        for l in self.p.stdout:
            self.q.put(l.rstrip("\n"))
        self.q.put(None)

    def w(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, pred, tmo, collect=None):
        t0 = time.time()
        while time.time() - t0 < tmo:
            try:
                l = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if l is None:
                raise RuntimeError("engine %s closed" % self.kind)
            if collect:
                collect(l)
            if pred(l):
                return l
        raise RuntimeError("timeout %s" % self.kind)

    def isready(self, first=False):
        t0 = time.time()
        self.w("isready")
        bad = []

        def col(l):
            if "P2NNUE loaded" in l or "loaded" in l.lower():
                self.loaded.append(l[:120])
            if ("MISMATCH" in l or "cannot open" in l.lower() or "error" in l.lower()) and "book" not in l.lower():
                bad.append(l[:160])
        self.wait(lambda l: l.startswith("readyok"), 300, col)
        if bad:
            raise RuntimeError("isready: " + " | ".join(bad))
        return time.time() - t0

    def search(self, position, nodes, fresh=True):
        tr = self.isready() if fresh else 0.0
        self.w("position " + position)
        self.w("go nodes %d" % nodes)
        last, last_exact = None, None
        t0 = time.time()
        while True:
            l = self.q.get(timeout=3600)
            if l is None:
                raise RuntimeError("closed")
            if l.startswith("info") and " score " in l:
                last = l
                if "lowerbound" not in l and "upperbound" not in l:
                    last_exact = l
            if l.startswith("bestmove"):
                bm = l.split()[1]
                break
        dt = time.time() - t0
        src = last_exact or last
        r = dict(bestmove=bm, sec=round(dt, 2), isready_sec=round(tr, 2), bound=(last_exact is None and last is not None))
        if src:
            t = src.split()
            for key in ("depth", "nodes", "seldepth"):
                if key in t:
                    try:
                        r[key] = int(t[t.index(key) + 1])
                    except ValueError:
                        pass
            if "pv" in t:
                r["pv"] = t[t.index("pv") + 1:]
            i = t.index("score")
            if t[i + 1] == "cp":
                r["v"] = int(t[i + 2])
            elif t[i + 1] == "mate":
                m = t[i + 2]
                r["mate"] = m
                r["v"] = -MATE if m.startswith("-") else MATE
        return r

    def close(self):
        try:
            self.w("quit")
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def load_sets():
    return [json.loads(l) for l in io.open(OUT + "/sets.jsonl", encoding="utf-8")]


def jobs_for(S, only=None):
    """優先順: a20 部分集合 (s1x3, b, a2, a20) -> 残り (s1x3, b, a2)。 a20 は長いので部分集合内でも最後。"""
    sub = [x for x in S if x["a20"]]
    rest = [x for x in S if not x["a20"]]
    J = []
    for x in sub:
        J += [(x["id"], "s1", r) for r in range(3)] + [(x["id"], "b", 0), (x["id"], "a2", 0)]
    J += [(x["id"], "a20", 0) for x in sub]
    if only != "a20sub":
        for x in rest:
            J += [(x["id"], "s1", r) for r in range(3)] + [(x["id"], "b", 0), (x["id"], "a2", 0)]
    return J


def done_set():
    d = set()
    if os.path.exists(RAW):
        for l in io.open(RAW, encoding="utf-8"):
            try:
                o = json.loads(l)
            except Exception:
                continue
            if "v" in o or "bestmove" in o:
                d.add((o["id"], o["cond"], o["rep"]))
    return d


def run(workers, only):
    S = load_sets()
    pos = dict((x["id"], x["position"]) for x in S)
    dn = done_set()
    J = [j for j in jobs_for(S, only) if j not in dn]
    print("  残りの測定 %d (済 %d) / workers %d" % (len(J), len(dn), workers))
    jq = queue.Queue()
    for j in J:
        jq.put(j)
    lock = threading.Lock()
    cnt = [0]
    t_start = time.time()

    def worker(wi):
        E = {}
        try:
            while True:
                try:
                    pid, cond, rep = jq.get_nowait()
                except queue.Empty:
                    return
                kind, nodes = COND[cond]
                if kind not in E:
                    E[kind] = Eng(kind)
                    with lock:
                        print("  [w%d] %s 起動 missing=%s loaded=%s" % (wi, kind, E[kind].missing, E[kind].loaded[:2]))
                try:
                    r = E[kind].search(pos[pid], nodes)
                except Exception as e:  # エンジンが落ちたら作り直して 1 回だけ再試行
                    with lock:
                        print("  [w%d] %s %s %s 失敗 %s -> 再起動" % (wi, pid, cond, rep, e))
                    try:
                        E[kind].close()
                    except Exception:
                        pass
                    E[kind] = Eng(kind)
                    r = E[kind].search(pos[pid], nodes)
                r.update(id=pid, cond=cond, rep=rep, req_nodes=nodes, worker=wi)
                with lock:
                    with io.open(RAW, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(r) + "\n")
                    cnt[0] += 1
                    if cnt[0] % 25 == 0:
                        el = time.time() - t_start
                        print("  %s 済 %d / %d  (%.0f 分経過)" % (time.strftime("%H:%M:%S"), cnt[0], len(J), el / 60))
        finally:
            for e in E.values():
                e.close()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print("  完了 %s" % time.strftime("%H:%M:%S"))


def test():
    S = load_sets()
    x = [s for s in S if s["set"] == "PRE"][0]
    for kind in ("ours", "v4"):
        e = Eng(kind)
        print(kind, "missing", e.missing, "loaded", e.loaded[:3])
        for nodes in (390000, 390000, 2000000):
            r = e.search(x["position"], nodes)
            print("  ", nodes, r)
        e.close()


# ------------------------------------------------------------------ 集計
def clipv(v, c=2000.0):
    return max(-c, min(c, v))


def boot_ratio(num, den, games, B=4000, seed=1):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(games, return_inverse=True)
    ns = np.bincount(inv, weights=num)
    ds = np.bincount(inv, weights=den)
    out = []
    for _ in range(B):
        s = rng.integers(0, len(ug), len(ug))
        d = ds[s].sum()
        out.append(ns[s].sum() / d if d != 0 else np.nan)
    return np.nanpercentile(out, [2.5, 97.5])


def cl_se(x, games):
    x = np.asarray(x, float)
    ug, inv = np.unique(games, return_inverse=True)
    mu = x.mean()
    sg = np.bincount(inv, weights=x - mu)
    return np.sqrt((sg ** 2).sum() * len(ug) / max(1, len(ug) - 1)) / len(x)


def kreg(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (x * y).sum() / (x * x).sum() if len(x) else np.nan


def win_game_k():
    """勝ち局の棋譜から k (v4 の値 = k x 我々の値、 原点回帰)。 我々 ply q と v4 ply q+1 (我々視点) の対。"""
    sys.path.insert(0, OUT)
    import make_sets as M
    pairs = collections.defaultdict(list)
    for fn, ours, opp, prio in M.FILES:
        for g in M.load(fn, ours, opp, prio):
            if g["outcome"] != "win":
                continue
            byp = dict((x["ply"], x) for x in g["seq"])
            for x in g["seq"]:
                y = byp.get(x["ply"] + 1)
                if not x["mine"] or y is None or x["ov"] is None or y["ov"] is None:
                    continue
                if abs(x["ov"]) > 1500 or abs(y["ov"]) > 1500:
                    continue
                pb = 0 if x["ply"] < 40 else 1 if x["ply"] < 80 else 2
                pairs[(g["src"] != "B40", pb)].append((x["ov"], y["ov"]))
    return pairs


def analyze():
    log = []

    def P(*a):
        s = " ".join(str(z) for z in a)
        print(s)
        log.append(s)
    S = dict((x["id"], x) for x in load_sets())
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    BM = collections.defaultdict(dict)
    meta = collections.defaultdict(list)
    for l in io.open(RAW, encoding="utf-8"):
        o = json.loads(l)
        if "v" not in o:
            continue
        R[o["id"]][o["cond"]].append(o["v"])
        if o["rep"] == 0:
            BM[o["id"]][o["cond"]] = o["bestmove"]
        meta[o["cond"]].append((o.get("nodes", 0), o.get("depth", 0), o["sec"], o.get("isready_sec", 0), o.get("bound")))
    P("=== P0 (DESIGN-20261112-OPTIMISM / 線は固定)")
    P("  現職 %s + %s / v4 %s" % (os.path.basename(OURS), NET.split("/")[-2], os.path.basename(V4)))
    for c in ("s1", "a2", "a20", "b"):
        m = meta[c]
        if m:
            a = np.array([x[:4] for x in m], float)
            P("  %-3s 測定 %4d / 実ノード中央 %9.0f / depth 中央 %4.1f / 探索 %.1f s / isready %.2f s / bound 行で読んだ %d"
              % (c, len(m), np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2]), np.median(a[:, 3]),
                 sum(1 for x in m if x[4])))

    def val(pid, c):
        v = R[pid].get(c)
        if not v:
            return None
        if c == "s1":
            return float(np.median(v)) if len(v) == 3 else None
        return float(v[0])

    rows = []
    for pid, x in S.items():
        d = dict(id=pid, set=x["set"], game=x["game"], ply=x["ply"], win=x.get("win"), a20sub=x["a20"],
                 played=x["played"], s_game=x["s_game"])
        for c in ("s1", "a2", "a20", "b"):
            d[c] = val(pid, c)
        d["s1reps"] = R[pid].get("s1", [])
        d["bm"] = BM[pid]
        rows.append(d)

    # ---- 雑音床 (OFF 対 OFF = s1 の 3 回どうし)
    P("")
    P("  --- 雑音床: s1 の 3 回どうしの差 (同一条件 / isready 済み)")
    for st in ("PRE", "NEG", "POS"):
        dd = []
        for d in rows:
            r3 = d["s1reps"]
            if d["set"] == st and len(r3) == 3 and max(abs(v) for v in r3) < 2000:
                dd += [r3[0] - r3[1], r3[0] - r3[2], r3[1] - r3[2]]
        if dd:
            dd = np.array(dd)
            P("  %s: 対 %d / 平均 %+.1f / SD %.1f / 平均|差| %.1f  -> 中央値 (3 回) の SD 目安 %.1f"
              % (st, len(dd), dd.mean(), dd.std(), np.abs(dd).mean(), dd.std() / np.sqrt(2) * 1.2533 / np.sqrt(3)))

    # ---- k
    P("")
    P("  --- 目盛り k (v4 の値 = k x 我々の値 / 原点回帰)")
    ks = {}
    pairs = win_game_k()
    for kh in (True, False):
        for pb in (0, 1, 2):
            p = np.array(pairs.get((kh, pb), []))
            if len(p):
                P("  勝ち局の棋譜 %-4s ply %-6s n %5d  k = %.3f" % ("KH" if kh else "B40", ("<40", "40-79", ">=80")[pb], len(p), kreg(p[:, 0], p[:, 1])))
    allp = np.array([v for (kh, pb), l in pairs.items() if kh for v in l])
    ks["勝ち局KH"] = kreg(allp[:, 0], allp[:, 1])
    allp2 = np.array([v for l in pairs.values() for v in l])
    P("  勝ち局 KH 全体 k = %.3f (n %d) / 3 ファイル全体 k = %.3f (n %d)" % (ks["勝ち局KH"], len(allp), kreg(allp2[:, 0], allp2[:, 1]), len(allp2)))
    neg = [d for d in rows if d["set"] == "NEG" and d["a2"] is not None and d["b"] is not None
           and abs(d["a2"]) < 2000 and abs(d["b"]) < 2000]
    for nm, f in (("NEG b~a2", lambda d: d["a2"]), ("NEG b~s1", lambda d: d["s1"])):
        for pb, lo, hi in ((0, 0, 40), (1, 40, 80), (2, 80, 999)):
            sel = [d for d in neg if lo <= d["ply"] < hi and f(d) is not None]
            if sel:
                P("  %s ply %3d-%3d n %3d  k = %.3f" % (nm, lo, hi, len(sel), kreg([f(d) for d in sel], [d["b"] for d in sel])))
    ks["NEG(b~a2)"] = kreg([d["a2"] for d in neg], [d["b"] for d in neg])
    P("  NEG 全体 b~a2 k = %.3f (n %d)  🚨NEG は ±150 の静かな局面なので k の分解能は低い" % (ks["NEG(b~a2)"], len(neg)))
    # PRE/POS の a2 対 b も参考 (同ノード数どうし)
    for st in ("PRE", "POS"):
        sel = [d for d in rows if d["set"] == st and d["a2"] is not None and d["b"] is not None and abs(d["a2"]) < 2000 and abs(d["b"]) < 2000]
        if sel:
            P("  (参考) %s b~a2 k = %.3f (n %d)" % (st, kreg([d["a2"] for d in sel], [d["b"] for d in sel]), len(sel)))

    # ---- 主判定 (PRE の a20 部分集合)
    P("")
    P("  --- 分岐 X/Y/Z: PRE a20 部分集合で 比 = mean(s1 - a20) / mean(s1 - b/k)  (値は ±2000 で clip)")
    pre = [d for d in rows if d["set"] == "PRE" and d["a20sub"] and None not in (d["s1"], d["a20"], d["b"])]
    nbig = sum(1 for d in pre if max(abs(d["s1"]), abs(d["a20"]), abs(d["b"])) >= 2000)
    P("  n %d (|値|>=2000 を含む局面 %d → clip)" % (len(pre), nbig))
    s1 = np.array([clipv(d["s1"]) for d in pre])
    a20 = np.array([clipv(d["a20"]) for d in pre])
    bb = np.array([clipv(d["b"]) for d in pre])
    gm = np.array([d["game"] for d in pre])
    deep = s1 - a20
    P("  s1 平均 %+.1f / a20 平均 %+.1f / b 平均 %+.1f" % (s1.mean(), a20.mean(), bb.mean()))
    P("  深さで見える量 mean(s1 - a20) = %+.1f cp (SE %.1f)" % (deep.mean(), cl_se(deep, gm)))
    verdicts = {}
    klist = [("1.19", 1.19), ("1.3", 1.3), ("1.43", 1.43)] + [(k, v) for k, v in ks.items()]
    for kn, k in klist:
        tru = s1 - bb / k
        ratio = deep.mean() / tru.mean() if tru.mean() != 0 else np.nan
        lo, hi = boot_ratio(deep, tru, gm)
        if tru.mean() <= 0:
            br = "— (本物の楽観が正でない)"
        else:
            br = "X" if ratio >= 0.40 else ("Y" if ratio >= 0.15 else "Z")
        verdicts[kn] = br
        sgn = np.mean(np.sign(deep) != np.sign(tru))
        P("  k=%-10s k=%.3f  本物の楽観 mean(s1-b/k) %+6.1f (SE %.1f) / 比 %.2f [%.2f, %.2f] -> %s / 符号不一致 %d/%d (%.0f%%)"
          % (kn, k, tru.mean(), cl_se(tru, gm), ratio, lo, hi, br, int(np.sum(np.sign(deep) != np.sign(tru))), len(pre), 100 * sgn))
    for w in ("A", "B"):
        m = np.array([d["win"] == w for d in pre])
        if m.any():
            tru = s1[m] - bb[m] / 1.19
            P("    窓 %s (T%s) n %d: s1-a20 %+.1f / s1-b/1.19 %+.1f / 比 %.2f"
              % (w, "-20..-11" if w == "A" else "-10..-2", m.sum(), deep[m].mean(), tru.mean(), deep[m].mean() / tru.mean() if tru.mean() else np.nan))
    # 参考: PRE 300 全体で a2
    pa = [d for d in rows if d["set"] == "PRE" and None not in (d["s1"], d["a2"], d["b"])]
    if pa:
        s1p = np.array([clipv(d["s1"]) for d in pa]); a2p = np.array([clipv(d["a2"]) for d in pa]); bp = np.array([clipv(d["b"]) for d in pa])
        for k in (1.19, 1.3, 1.43):
            P("  (参考) PRE %d 全体 a2: mean(s1-a2) %+.1f / mean(s1-b/%.2f) %+.1f / 比 %.2f"
              % (len(pa), (s1p - a2p).mean(), k, (s1p - bp / k).mean(), (s1p - a2p).mean() / (s1p - bp / k).mean()))

    # ---- 器の検算
    P("")
    P("  --- 器の検算")
    chk = {}
    for st in ("POS", "NEG"):
        sel = [d for d in rows if d["set"] == st and d["a20sub"] and None not in (d["s1"], d["a20"])]
        if not sel:
            continue
        df = np.array([clipv(d["s1"]) - clipv(d["a20"]) for d in sel])
        g2 = [d["game"] for d in sel]
        P("  %s n %d: mean(s1 - a20) %+.1f (SE %.1f) / mean|s1 - a20| %.1f / 中央 %+.1f"
          % (st, len(sel), df.mean(), cl_se(df, g2), np.abs(df).mean(), np.median(df)))
        if st == "POS":
            chk["POS"] = df.mean() >= 100
            P("    POS: s1 - a20 >= 100 %s" % ("✅" if chk["POS"] else "❌"))
        else:
            chk["NEG"] = abs(df.mean()) <= 30
            P("    NEG: |mean(s1 - a20)| <= 30 %s  (平均絶対差 %.1f <= 30 なら %s)"
              % ("✅" if chk["NEG"] else "❌", np.abs(df).mean(), "✅" if np.abs(df).mean() <= 30 else "❌"))
            sel2 = [d for d in rows if d["set"] == "NEG" and None not in (d["s1"], d["b"])]
            if sel2:
                t = np.array([d["s1"] - d["b"] / 1.19 for d in sel2])
                P("    NEG: mean(s1 - b/1.19) %+.1f (n %d) = 陰性対照で本物の楽観が 0 付近か" % (t.mean(), len(t)))

    # ---- 一致率
    P("")
    P("  --- a20 の bestmove と実戦手の一致率 (参考 s1 rep0 / a2 / b)")
    for st in ("PRE", "NEG", "POS"):
        out = []
        for c in ("a20", "s1", "a2", "b"):
            sel = [d for d in rows if d["set"] == st and c in d["bm"] and (c != "a20" or d["a20sub"])]
            if st and sel:
                out.append("%s %d/%d (%.0f%%)" % (c, sum(d["bm"][c] == d["played"] for d in sel), len(sel),
                                                  100.0 * np.mean([d["bm"][c] == d["played"] for d in sel])))
        P("  %s: %s" % (st, " / ".join(out)))
        sel = [d for d in rows if d["set"] == st and "a20" in d["bm"] and "s1" in d["bm"]]
        if sel:
            P("     a20 と s1 rep0 の手の一致 %.0f%% (n %d)" % (100 * np.mean([d["bm"]["a20"] == d["bm"]["s1"] for d in sel]), len(sel)))

    P("")
    P("=== 判定")
    P("  器の検算: POS %s / NEG %s" % ("✅" if chk.get("POS") else "❌", "✅" if chk.get("NEG") else "❌"))
    P("  分岐: %s" % ", ".join("k=%s -> %s" % (k, v) for k, v in verdicts.items()))
    io.open("C:/project/p2shogi/runs/prereg/VERDICT-OPTIMISM-P0-20261112.log", "w", encoding="utf-8").write("\n".join(log) + "\n")
    with io.open(OUT + "/p0_table.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        run(w, only)
    elif cmd == "analyze":
        analyze()
    elif cmd == "test":
        test()
    else:
        print(__doc__)
