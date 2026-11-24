# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""V4CONFIRM (2026-11-16): 公開する版の「対 NAGISA v4」の強さを時間制で測る器。
事前登録 = runs/prereg/PREREG-20261116-V4CONFIRM.md (結果を見る前に固定)。

枠は 11-09 の B40 (対 v4 40.0% / 845 局) と同じ:
  tournament.exe / byoyomi 1000 / Threads 1 / Hash 512 / max_moves 512 / timeout margin 5000 ms /
  concurrency 2 / 定跡なし / ND=ND2=120 / MinimumThinkingTime 1000 / RoundUpToFullSecond false /
  --startpos-sequential (1 局面 x 先後) / 相手 v4 の option 8 本も B40 と同値。
我々の option = 本番 toml (p2shogi_denryusen_20261114.toml) から B40 と同じ鍵を落とし、
  EvalFile を L1 に、 KHFile と OPN の 7 鍵を足したもの。 B40 の option との差が
  「EvalFile / KHFile / OPN_*」だけであることを起動前に照合する (違えば止める)。

usage:
  py -3 v4c.py sets                              確認対局の 600 局面 (27 + 50k 行) と仮対局の 25 局面を作る (上書きしない)
  py -3 v4c.py calib <label> <opn quantised.bin> 最終の OPN net の目盛り (1/SD 比 x1024) を測り opn_<label>.json に書く
  py -3 v4c.py check <opn.json>                  option を組み、 両 engine を起動して読み込み行を照合 (対局しない)
  py -3 v4c.py run <opn.json> <set> <OUT_LABEL>  照合 -> tournament.exe。 set = confirm600 | prov25
                                                 出力 runs/selfplay/<OUT_LABEL>/ (既にあれば止める = rm しない)
  py -3 v4c.py verdict <out_dir>                 事前登録の判定 (満了後) / 400 局時点の打ち切り線の報告
"""
import collections, io, json, math, os, queue, re, subprocess, sys, threading, time, tomllib

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HERE = "C:/project/p2shogi/runs/v4confirm"
ROOT = "C:/project/p2shogi"
TOURN = ROOT + "/target/release/tournament.exe"
SRC = ROOT + "/data/startpos/start_sfens_ply24.txt"
TOML = ROOT + "/p2shogi_denryusen_20261114.toml"
B40META = ROOT + "/runs/selfplay/B40/meta.json"
USED_SETS = [ROOT + "/runs/opnkpi/sets.jsonl", ROOT + "/runs/selfplay/OPNKPI2_20261116/sets.jsonl"]

EXE = "E:/shogi/p2shogi_cplus/release/opn/tsumugi_opnkh_1115r2_pgo.exe"
L1NET = "E:/shogi/p2shogi/tmp_mob/khhr/a-4/quantised.bin"
KHFILE = "E:/shogi/p2shogi/tmp_mob/khhr/kh_final.bin"
OPN_FIXED = [("OPN_LambdaMax", "256"), ("OPN_ProgHi", "-1199"), ("OPN_ProgWidth", "100")]

V4EXE = "E:/shogi/engines/nagisa_v4_20261101/NAGISA_V4-Windows-AVX2/NAGISA_V4.exe"
V4OPTS = [("EvalDir", "E:/shogi/engines/nagisa_v4/NAGISA_V4-Windows-AVX2/eval"), ("FV_SCALE", "28"),
          ("USI_OwnBook", "false"), ("USI_Ponder", "false"), ("NetworkDelay", "120"), ("NetworkDelay2", "120"),
          ("MinimumThinkingTime", "1000"), ("RoundUpToFullSecond", "false")]
LBL_OURS, LBL_V4 = "ours", "nag4"

# B40 と同じく落とす鍵 (定跡 / ファイル / 対局器が持つ設定)。 BookExit* と BookFile3 は B40 の後に toml に入った定跡系
DROP = {"BookFile", "BookFile2", "BookFile3", "BookDir", "BookOnTheFly", "USI_OwnBook", "IgnoreBookPly",
        "BookMoves", "BookDepthLimit", "BookEvalBlackLimit", "BookEvalWhiteLimit", "BookEvalDiff", "NarrowBook",
        "ConsiderBookMoveCount", "BookIgnoreRate", "BookPvMoves", "FlippedBook", "BookRepetitionGuard",
        "BookExitFile", "BookExitMargin", "BookExitMaxPly",
        "EvalFile", "EvalDir", "Threads", "USI_Hash", "NetworkDelay", "NetworkDelay2", "MinimumThinkingTime",
        "MaxMovesToDraw", "USI_Ponder"}
FRAME = [("EvalFile", L1NET), ("KHFile", KHFILE), ("USI_OwnBook", "false"), ("USI_Ponder", "false"),
         ("NetworkDelay", "120"), ("NetworkDelay2", "120"), ("MinimumThinkingTime", "1000"),
         ("RoundUpToFullSecond", "false")]

SETS = {  # 名前 -> (ファイル, 開始行, 間隔, 個数)
    "confirm600": (HERE + "/sets_confirm600.jsonl", 27, 50, 600),   # 27/77 mod 100 = 100 行おき (打ち切り対局) と重ならない、 5 の倍数でないので 45 行おき (B40) とも重ならない
    "c50": (HERE + "/sets_c50.jsonl", 27, 50, 50),
    "prov25": (HERE + "/sets_prov25.jsonl", 13, 1200, 25),          # 13 mod 50 なので confirm600 とも重ならない (仮対局用)
}


def log(path, msg):
    s = "%s %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(s)
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")


# ---------------------------------------------------------------- 局面
def cmd_sets():
    L = [l.strip() for l in io.open(SRC, encoding="utf-8") if l.strip()]
    used_lines, used_sfen = set(), set()
    for p in USED_SETS:
        for l in io.open(p, encoding="utf-8"):
            o = json.loads(l)
            used_lines.add(o["line"]); used_sfen.add(o["sfen"])
    b40 = set(s.split("sfen ", 1)[1] for s in json.load(io.open(B40META, encoding="utf-8"))["start_positions"])
    picked = {}
    for name, (path, off, step, n) in SETS.items():
        rows = [dict(pos=j, line=off + j * step, sfen=L[off + j * step][5:]) for j in range(n)]
        assert all(L[r["line"]].startswith("sfen ") for r in rows)
        hit_l = sum(r["line"] in used_lines for r in rows)
        hit_s = sum(r["sfen"] in used_sfen for r in rows)
        hit_b = sum(r["sfen"] in b40 for r in rows)
        dup = n - len(set(r["sfen"] for r in rows))
        print("%s: %d 局面 (行 %d + %d k / 最終行 %d) 打ち切り対局と 行一致 %d / sfen 一致 %d / B40 と sfen 一致 %d / 集合内重複 %d"
              % (name, n, off, step, rows[-1]["line"], hit_l, hit_s, hit_b, dup))
        if hit_l or hit_s or dup:
            raise SystemExit("重なりがある: " + name)
        picked[name] = rows
    a = set(r["sfen"] for r in picked["confirm600"]); b = set(r["sfen"] for r in picked["prov25"])
    print("confirm600 と prov25 の sfen 一致 %d" % len(a & b))
    if a & b:
        raise SystemExit("confirm600 と prov25 が重なる")
    for name, rows in picked.items():
        path = SETS[name][0]
        txt = path.replace(".jsonl", ".txt")
        if os.path.exists(path) or os.path.exists(txt):
            print("既にある (上書きしない): " + path); continue
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with io.open(txt, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write("position sfen %s\n" % r["sfen"])
        print("書いた " + path + " / " + txt)


# ---------------------------------------------------------------- 目盛り
def cmd_calib(label, opn):
    """screen_<label>.jsonl (篩のキャッシュ) が無ければ作り、 calib_cov.py の 1/SD 比 (bucket 0-2) x 1024 を目盛りにする。"""
    import numpy as np
    V4E = "E:/shogi/p2shogi/tmp_mob/v4evalana/"
    scr = V4E + "screen_%s.jsonl" % label
    if not os.path.exists(scr):
        print("篩のキャッシュが無いので作る: " + scr)
        subprocess.run(["py", "-3", ROOT + "/runs/opnarch/opn_static2.py", "net", label, opn], check=True)
    out = subprocess.run(["py", "-3", ROOT + "/runs/opnint/calib_cov.py", label], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=True, cwd=ROOT + "/runs/opnint").stdout
    print(out)
    printed = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"b(\d) n .*?1/SD比 ([0-9.]+) / 現職", out)}
    # 3 桁に丸められているので、 同じ選別で SD 比を自前で精密に出し、 印字値と照合する
    R = [json.loads(l) for l in io.open(V4E + "m2_games.jsonl", encoding="utf-8")]
    for r, l in zip(R, io.open(scr, encoding="utf-8")):
        r["sx"] = json.loads(l)["s"]
    so, sx, bk = [], [], []
    for r in R:
        if r["incheck"] or max(abs(r["so"]), abs(r["sv"]), abs(r["sx"])) > 1500:
            continue
        sgn = 1.0 if r["mover"] == "ours" else -1.0
        so.append(sgn * r["so"]); sx.append(sgn * r["sx"]); bk.append(r["bucket"])
    so, sx, bk = map(np.array, (so, sx, bk))
    sc = []
    for b in (0, 1, 2):
        m = bk == b
        k = so[m].std() / sx[m].std()
        if abs(k - printed[b]) > 0.0006:
            raise SystemExit("calib_cov.py の印字 %.3f と自前 %.5f が合わない (b%d)" % (printed[b], k, b))
        sc.append(int(round(k * 1024)))
        print("  b%d 1/SD 比 %.5f -> Scale%d = %d" % (b, k, b, sc[-1]))
    path = HERE + "/opn_%s.json" % label
    if os.path.exists(path):
        old = json.load(io.open(path, encoding="utf-8"))
        if old["scale"] != sc or old["opnfile"] != opn:
            raise SystemExit("既にある %s と値が違う (上書きしない): %s" % (path, old))
        print("既にある (同値): " + path); return
    json.dump(dict(label=label, opnfile=opn, scale=sc, measured=time.strftime("%Y-%m-%d %H:%M:%S")),
              io.open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("書いた " + path)


# ---------------------------------------------------------------- option
def fmt(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def our_exe(opn):
    return opn.get("exe", EXE)


def our_options(opn):
    t = tomllib.load(open(TOML, "rb"))
    base = [(k, fmt(v)) for k, v in t["engine"]["options"].items() if k not in DROP]
    opts = base + FRAME
    fixed = [tuple(x) for x in opn.get("fixed", OPN_FIXED)]
    # 11-18: OPN は bucket 0〜1 だけ (OPN_Scale2 は exe から削除)。 既存の json の scale は 3 要素のままなので先頭 2 つだけ使う。
    #   json に "offset" (cp) があれば OPN_Offset を渡す (無ければ渡さない = exe の既定 0)。
    opts += [("OPNFile", opn["opnfile"])] + fixed + [("OPN_Scale%d" % i, str(s)) for i, s in enumerate(opn["scale"][:2])]
    if "offset" in opn:
        opts += [("OPN_Offset", str(int(opn["offset"])))]
    # 改善案の option (既にある鍵は値を差し替え、 無い鍵は足す)
    extra = [tuple(x) for x in opn.get("extra", [])]
    ex = dict(extra)
    opts = [(k, ex.pop(k) if k in ex else v) for k, v in opts] + [(k, v) for k, v in extra if k in ex]
    keys = [k for k, _ in opts]
    assert len(keys) == len(set(keys)), "鍵の重複"
    # B40 の option との照合: 差は EvalFile / KHFile / OPN_* だけ
    b40 = dict(x.split("=", 1) for x in json.load(io.open(B40META, encoding="utf-8"))["engines"][0]["usi_options"])
    d = dict(opts)
    allowed = {"EvalFile", "KHFile", "OPNFile"} | {k for k in d if k.startswith("OPN_")} | {k for k, _ in opn.get("extra", [])}
    bad = [k for k in set(d) | set(b40) if k not in allowed and d.get(k) != b40.get(k)]
    if bad:
        raise SystemExit("B40 と想定外の差: %s" % {k: (d.get(k), b40.get(k)) for k in bad})
    return opts, {k: (b40.get(k), d.get(k)) for k in sorted(allowed) if d.get(k) != b40.get(k)}


def expect_tail(opn):
    f = dict(tuple(x) for x in opn.get("fixed", OPN_FIXED))   # 11-16: 改善案で ProgHi などを変えたときも照合が合うように
    # 11-18: 新しい exe は "scale=a,b offset_cp=X active=1"、 旧 exe は "scale=a,b,c active=1" と出す。 先頭 2 つまでの部分一致で両方に合う。
    return "lam_max=%s prog_hi=%s width=%s scale=%d,%d" % (
        f["OPN_LambdaMax"], f["OPN_ProgHi"], f["OPN_ProgWidth"], *tuple(opn["scale"][:2]))


class Usi:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe], cwd=os.path.dirname(exe), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()

    def _rd(self):
        for l in self.p.stdout:
            self.q.put(l.rstrip("\r\n"))
        self.q.put(None)

    def w(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def until(self, pred, tmo):
        out, t0 = [], time.time()
        while time.time() - t0 < tmo:
            try:
                l = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if l is None:
                raise RuntimeError("engine が閉じた: %s" % out[-5:])
            out.append(l)
            if pred(l):
                return out
        raise RuntimeError("timeout: %s" % out[-5:])

    def boot(self, opts):
        self.w("usi")
        names = set()
        for l in self.until(lambda l: l == "usiok", 60):
            m = re.match(r"option name (\S+) type", l)
            if m:
                names.add(m.group(1))
        missing = [k for k, _ in opts if k not in names and k not in ("USI_Hash", "Threads")]
        self.w("setoption name USI_Hash value 512"); self.w("setoption name Threads value 1")
        for k, v in opts:
            self.w("setoption name %s value %s" % (k, v))
        self.w("isready")
        lines = self.until(lambda l: l == "readyok", 300)
        self.w("quit")
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        return missing, lines


def check(opn, logpath):
    opts, diff = our_options(opn)
    log(logpath, "我々 %s / option %d 本 (B40 との差: %s)" % (our_exe(opn), len(opts), diff))
    for f in (our_exe(opn), dict(our_options(opn)[0]).get("EvalFile", L1NET), KHFILE, opn["opnfile"], V4EXE, V4OPTS[0][1]):
        if not os.path.exists(f):
            raise SystemExit("無い: " + f)
    missing, lines = Usi(our_exe(opn)).boot(opts)
    keep = [l for l in lines if re.search(r"loaded|P2OPN|P2NNUE|KH|error|Error|cannot|MISMATCH", l)]
    for l in keep:
        log(logpath, "  [ours] " + l[:400])
    errs = []
    if missing:
        errs.append("exe に無い option: %s" % missing)
    if any(re.search(r"error|cannot open|MISMATCH", l, re.I) and "book" not in l.lower() for l in lines):
        errs.append("エラー行がある")
    evf = dict(opts).get("EvalFile", L1NET)   # 11-16: 改善案で EvalFile を差し替える (L1Q128) ので実際に渡した値で照合
    if not any("P2NNUE loaded" in l and evf in l for l in lines):
        errs.append("P2NNUE の行に EvalFile (%s) が無い" % evf)
    if not any("KH: loaded" in l and KHFILE in l for l in lines):
        errs.append("KH の読み込み行が無い")
    if not any("P2OPN loaded" in l and opn["opnfile"] in l and expect_tail(opn) in l for l in lines):
        errs.append("P2OPN の行が期待値 (%s / %s) と違う" % (opn["opnfile"], expect_tail(opn)))
    vmiss, vlines = Usi(V4EXE).boot(V4OPTS)
    for l in vlines:
        if re.search(r"loaded|eval|Eval|error", l):
            log(logpath, "  [v4] " + l[:300])
    if vmiss:
        errs.append("v4 に無い option: %s" % vmiss)
    if errs:
        for e in errs:
            log(logpath, "  NG " + e)
        raise SystemExit("照合で止めた (対局しない)")
    log(logpath, "  照合 ok (P2NNUE=L1 / KH / P2OPN %s %s / v4 option 全部あり)" % (os.path.basename(os.path.dirname(opn["opnfile"])), expect_tail(opn)))
    return opts


def cmd_check(opnjson):
    check(json.load(io.open(opnjson, encoding="utf-8")), HERE + "/check.log")


# ---------------------------------------------------------------- 対局
def cmd_run(opnjson, setname, label):
    opn = json.load(io.open(opnjson, encoding="utf-8"))
    out = ROOT + "/runs/selfplay/" + label
    if os.path.exists(out):
        raise SystemExit("出力先が既にある (同じラベルで再起動しない / 消さない): " + out)
    spath = SETS[setname][0].replace(".jsonl", ".txt")
    npos = sum(1 for l in io.open(spath, encoding="utf-8") if l.strip())
    os.makedirs(out)
    L = out + "/chain.log"
    log(L, "=== V4CONFIRM run %s (局面 %s = %d 件 x 先後 = %d 局 / OPN %s)" % (label, setname, npos, 2 * npos, opnjson))
    opts = check(opn, L)
    args = [TOURN, "--engine", our_exe(opn), "--engine", V4EXE, "--engine-label", LBL_OURS, "--engine-label", LBL_V4,
            "--base-label", LBL_OURS, "--games", str(npos), "--concurrency", "2", "--byoyomi", "1000",
            "--threads", "1", "--hash-mb", "512", "--max-moves", "512", "--timeout-margin-ms", "5000",
            "--startpos-file", spath, "--startpos-sequential", "--report-interval", "10", "--out-dir", out]
    for k, v in opts:
        args += ["--engine-usi-option", "0:%s=%s" % (k, v)]
    for k, v in V4OPTS:
        args += ["--engine-usi-option", "1:%s=%s" % (k, v)]
    json.dump(dict(label=label, set=setname, startpos_file=spath, npos=npos, opn=opn, args=args),
              io.open(out + "/v4c_run.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tl = io.open(out + "/tournament_--engine.log", "w", encoding="utf-8")
    p = subprocess.Popen(args, cwd=ROOT, stdout=tl, stderr=subprocess.STDOUT)
    log(L, "  tournament.exe PID %d 起動 (この python の PID %d)" % (p.pid, os.getpid()))
    jl = out + "/%s-vs-%s.jsonl" % (LBL_OURS, LBL_V4)
    reported = False
    while p.poll() is None:
        time.sleep(60)
        if not reported and 2 * npos >= 1200 and os.path.exists(jl):
            n = sum(1 for l in io.open(jl, encoding="utf-8", errors="replace") if '"type":"result"' in l[:40])
            if n >= 400:
                reported = True
                v = summarize(jl, first=400)
                msg = "  [400 局の打ち切り線 / 報告のみ・止めない] 対応ペア %d 組 点推定 %.1f%% (2SE %.1f) -> %s" % (
                    v["pairs"], v["pm"], v["se2"], "45%% 未満 = 事前登録の打ち切り線に掛かった (止めない / メインに報告)" if v["pm"] < 45 else "45% 以上")
                log(L, msg)
                io.open(out + "/interim400.txt", "w", encoding="utf-8").write(msg + "\n")
    log(L, "  tournament.exe rc=%s" % p.returncode)
    subprocess.run(["py", "-3", ROOT + "/runs/rescore/jsonl_to_csa.py", repaired_dir(out), out + "/csa"],
                   stdout=io.open(out + "/csa_export.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    cmd_verdict(out)
    log(L, "=== end")


# ---------------------------------------------------------------- 集計
def read_records(jl):
    """jsonl を 1 レコードずつ返す。 11-16 の仮対局で result 行が 1 本途中で切れて次の move 行と繋がった
    (原因不明 / 13 byte 欠落) ので、 '{"type":' の境目で切り直し、 壊れた result は正規表現で鍵だけ拾う。
    戻り値 (records, 修復した数)。"""
    txt = io.open(jl, encoding="utf-8", errors="replace").read()
    parts = re.split(r'(?=\{"type":")', txt)
    recs, fixed = [], 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            recs.append(json.loads(p)); continue
        except ValueError:
            pass
        fixed += 1
        if p.startswith('{"type":"result"'):
            d = {"type": "result", "recovered": True}
            for k in ("game_id", "plies", "ticket_id", "pair_index", "pair_slot", "attempt", "startpos_idx"):
                m = re.search(r'"%s":(\d+)[,}]' % k, p)
                if m:
                    d[k] = int(m.group(1))
            for k in ("outcome", "reason", "winner"):
                m = re.search(r'"%s":"([^"]*)"' % k, p)
                if m:
                    d[k] = m.group(1)
            if "startpos_idx" not in d and "pair_index" in d:
                d["startpos_idx"] = d["pair_index"]   # --startpos-sequential では pair_index == startpos_idx (B40 で全件確認)
            if all(k in d for k in ("game_id", "outcome", "startpos_idx")):
                recs.append(d)
            else:
                print("  修復できない result: " + p[:200])
        else:
            print("  捨てた壊れ行: " + p[:200])
    return recs, fixed


def summarize(jl, first=None):
    recs, _ = read_records(jl)
    res = [r for r in recs if r.get("type") == "result"]
    tout = {r["game_id"] for r in recs if r.get("type") == "move" and r.get("timed_out")}
    # game_id はファイルごとの連番なので、 この 1 ファイルの中でだけ鍵に使う (腕は 1 本 / 鍵 = 局面 x 色)
    if first:
        res = res[:first]
    sc = {r["game_id"]: r for r in res if r["game_id"] not in tout}
    col = game_colors(jl)
    byk, wld, bycol = collections.defaultdict(dict), [0, 0, 0], {"b": [0, 0, 0], "w": [0, 0, 0]}
    for g, r in sc.items():
        c = col[g]
        o = r["outcome"]
        s = 0.5 if o == "draw" else (1.0 if (o == "black_win") == (c == "b") else 0.0)
        i = 0 if s == 1 else (1 if s == 0 else 2)
        wld[i] += 1; bycol[c][i] += 1
        byk[r["startpos_idx"]][c] = s
    ps = [(d["b"] + d["w"]) / 2 for d in byk.values() if "b" in d and "w" in d]
    n = len(sc)
    tot = (wld[0] + 0.5 * wld[2]) / n if n else float("nan")
    if len(ps) >= 2:
        pm = sum(ps) / len(ps); pse = math.sqrt(sum((x - pm) ** 2 for x in ps) / (len(ps) - 1) / len(ps))
    else:
        pm, pse = float("nan"), float("nan")
    return dict(n=n, timeouts=len(tout), wld=wld, bycol=bycol, total=100 * tot, pairs=len(ps), pm=100 * pm, se2=200 * pse)


def game_colors(jl):
    """game_id -> 我々の色。 1 手目の move 行の engine と side_to_move から決める。"""
    col = {}
    for d in read_records(jl)[0]:
        if d.get("type") == "move" and d.get("ply") == 1:
            stm = d["side_to_move"]
            col[d["game_id"]] = stm if d["engine"] == LBL_OURS else ("w" if stm == "b" else "b")
    return col


def repaired_dir(out):
    """壊れ行があれば修復した jsonl を <out>/repaired/ に書いてその dir を返す (元は触らない)。 無ければ out。"""
    jl = out + "/%s-vs-%s.jsonl" % (LBL_OURS, LBL_V4)
    recs, fixed = read_records(jl)
    if not fixed:
        return out
    rd = out + "/repaired"
    os.makedirs(rd, exist_ok=True)
    with io.open(rd + "/%s-vs-%s.jsonl" % (LBL_OURS, LBL_V4), "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(out + "/chain.log", "  jsonl に壊れ行 %d 本 -> 修復版を %s に書いた (元は触らない)" % (fixed, rd))
    return rd


def cmd_verdict(out):
    jl = out + "/%s-vs-%s.jsonl" % (LBL_OURS, LBL_V4)
    v = summarize(jl)
    L = out + "/verdict.txt"
    lines = ["===== V4CONFIRM 集計 %s (%s)" % (out, time.strftime("%m-%d %H:%M"))]
    w, l_, d = v["wld"]
    lines.append("  %d 局 (時間切れで除外 %d)  W-L-D %d-%d-%d  得点率 %.1f%%" % (v["n"], v["timeouts"], w, l_, d, v["total"]))
    for c, nm in (("b", "先手"), ("w", "後手")):
        a = v["bycol"][c]; m = sum(a)
        if m:
            lines.append("    %s %3d 局 %d-%d-%d  %.1f%%" % (nm, m, a[0], a[1], a[2], 100 * (a[0] + 0.5 * a[2]) / m))
    lines.append("  対応ペア %d 組  点推定 %.1f%%  2SE %.1f pt  区間 %.1f - %.1f%%" % (
        v["pairs"], v["pm"], v["se2"], v["pm"] - v["se2"], v["pm"] + v["se2"]))
    if v["pairs"] >= 600:
        tag = "互角以上" if v["pm"] >= 50.0 else ("互角と区別できない" if v["pm"] >= 47.4 else "v4 に届かない")
        lines.append("  事前登録の表記: **%s** (点推定 50.0%% 以上 = 互角以上 / 47.4-50.0 = 互角と区別できない / 未満 = v4 に届かない)" % tag)
    else:
        lines.append("  (600 組に満たないので事前登録の表記は付けない)")
    txt = "\n".join(lines)
    print(txt)
    io.open(L, "a", encoding="utf-8").write(txt + "\n")
    r = subprocess.run(["py", "-3", ROOT + "/runs/rescore/nobook55_verdict.py", repaired_dir(out) + "/%s-vs-%s.jsonl" % (LBL_OURS, LBL_V4), "--ours", LBL_OURS, "--target", "50"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    io.open(out + "/verdict_detail.txt", "w", encoding="utf-8").write(r.stdout + r.stderr)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    {"sets": lambda: cmd_sets(), "calib": lambda: cmd_calib(a[1], a[2]), "check": lambda: cmd_check(a[1]),
     "run": lambda: cmd_run(a[1], a[2], a[3]), "verdict": lambda: cmd_verdict(a[1])}[a[0]]()
