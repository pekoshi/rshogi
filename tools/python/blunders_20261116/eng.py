# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""11-16 BLUNDERS 用の USI 駆動 (runs/optimism/p0.py の Eng を、 仮対局 V4PROV50 の設定で使う形に直したもの)。
   - 我々の option は 仮対局の meta.json から取る (時間・定跡・Threads/Hash 系だけ差し替え)
   - 測定ごとに isready (TT/history を消す。 usinewgame は no-op)
   - bestmove を読んでから次のコマンド
   - BELOW_NORMAL 優先度"""
import io, json, os, queue, subprocess, sys, threading, time
META = "C:/project/p2shogi/runs/selfplay/V4PROV50_20261116/meta.json"
DROPK = ("BookFile", "BookDir", "BookOnTheFly", "USI_OwnBook", "Threads", "USI_Hash", "NetworkDelay", "NetworkDelay2",
         "MinimumThinkingTime", "MaxMovesToDraw", "USI_Ponder", "RoundUpToFullSecond", "PvInterval", "MultiPV")
MATE = 30000


def load(kind, hash_mb=512):
    m = json.load(io.open(META, encoding="utf-8"))
    e = [x for x in m["engines"] if x["label"] == ("ours" if kind == "ours" else "nag4")][0]
    o = []
    for kv in e["usi_options"]:
        k, _, v = kv.partition("=")
        if k in DROPK:
            continue
        o.append((k, v))
    o += [("USI_OwnBook", "false"), ("Threads", "1"), ("USI_Hash", str(hash_mb)), ("PvInterval", "0"),
          ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MinimumThinkingTime", "1")]
    return e["path"], o


class Eng:
    def __init__(self, kind, extra=(), hash_mb=512):
        self.kind = kind
        exe, opts = load(kind, hash_mb)
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1, cwd=os.path.dirname(exe), creationflags=0x40)  # IDLE_PRIORITY_CLASS (11-16 メイン指示: 時間制の仮対局と共存)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.w("usi")
        names = set()
        self.wait(lambda l: l.startswith("usiok"), 60, lambda l: names.add(l.split()[2]) if l.startswith("option name") else None)
        opts = opts + list(extra)
        self.missing = [k for k, _ in opts if k not in names]
        for k, v in opts:
            if k in names:
                self.w("setoption name %s value %s" % (k, v))
        self.isready()

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

    def isready(self):
        bad = []
        self.w("isready")
        self.wait(lambda l: l.startswith("readyok"), 300,
                  lambda l: bad.append(l[:160]) if (("MISMATCH" in l or "cannot open" in l.lower() or "error" in l.lower()) and "book" not in l.lower()) else None)
        if bad:
            raise RuntimeError("isready: " + " | ".join(bad))

    @staticmethod
    def parse(l):
        t = l.split()
        r = {}
        for key in ("depth", "nodes", "seldepth", "multipv"):
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
        else:
            s = t[i + 2]
            n = int(s.replace("+", "").replace("-", "") or 0)
            r["mate"] = -n if s.startswith("-") else n
            r["v"] = -(MATE - n) if s.startswith("-") else (MATE - n)
        return r

    def search(self, position, go, fresh=True, multipv=1):
        """go = 'nodes N' / 'byoyomi 1000' 等。 返り値: bestmove, v (手番視点), mate, depth, nodes, pv, lines (multipv 時の最終反復)"""
        if fresh:
            self.isready()
        self.w("position " + position)
        self.w("go " + go)
        last, last_exact, lines = None, None, {}
        t0 = time.time()
        while True:
            l = self.q.get(timeout=7200)
            if l is None:
                raise RuntimeError("closed")
            if l.startswith("info") and " score " in l and " pv " in l:
                bound = ("lowerbound" in l or "upperbound" in l)
                if multipv > 1:
                    if not bound:
                        d = self.parse(l)
                        lines[d.get("multipv", 1)] = d
                else:
                    last = l
                    if not bound:
                        last_exact = l
            if l.startswith("bestmove"):
                bm = l.split()[1]
                break
        r = dict(bestmove=bm, sec=round(time.time() - t0, 2))
        src = last_exact or last
        if src:
            r.update(self.parse(src))
            r["bound"] = last_exact is None
        if multipv > 1:
            r["lines"] = [lines[k] for k in sorted(lines)]
        return r

    def close(self):
        try:
            self.w("quit")
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
