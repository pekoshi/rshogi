# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""lzB (段 2/3) の NPS A/B (lzA_20261116/nps_ab.py の写し、 中身同じ)。 元: lzA 段 1 の NPS A/B。 runs/rescore/nps_cputime_ab.py をそのまま使い、 次の 2 点だけ足す。
   1. 両腕に本番の KHFile と OPN 一式 (it04_infrep.json の opnfile / fixed / scale) を --opt-a/--opt-b で渡す
   2. 11-12 NPSBUNDLE が落ちた OSError (Errno 22 = 起動直後のパイプ破損) は、 その 1 回の bench だけ 3 回まで取り直す
      (計測は readyok の後から。 取り直しは CPU 時間の比に偏りを入れない)
usage: py -3 -u nps_ab.py <A.exe> <B.exe> <周> <nodes>"""
import os
import io, json, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rescore"))
J = json.load(io.open("C:/project/p2shogi/runs/v4confirm/it04_infrep.json", encoding="utf-8"))
KH = "E:/shogi/p2shogi/tmp_mob/khhr/kh_final.bin"
NET = [v for k, v in J["extra"] if k == "EvalFile"][0]
o = ["KHFile=" + KH, "OPNFile=" + J["opnfile"]] + ["%s=%s" % (k, v) for k, v in J["fixed"]] + \
    ["OPN_Scale%d=%d" % (i, s) for i, s in enumerate(J["scale"])]
OPT = ";".join(o)
a, b, rounds, nodes = sys.argv[1:5]
sys.argv = [sys.argv[0], a, b, rounds, nodes, NET, "--opt-a", OPT, "--opt-b", OPT]
import nps_cputime_ab as M  # noqa: E402
_orig = M.bench_one
def bench_retry(*args, **kw):
    for k in range(3):
        try:
            return _orig(*args, **kw)
        except OSError as e:
            print("  (取り直し %d: %s)" % (k + 1, e), flush=True)
            time.sleep(2)
    return _orig(*args, **kw)
M.bench_one = bench_retry
print("  opt (両腕) = " + OPT)
M.main()
