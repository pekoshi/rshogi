# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""11-17 B2KH (ANALYSIS-20261117-BUCKET2-FABLE.md 4.2 の 1-2): KHFile の重みを a 倍した写しを作る。 元は書き換えない。
   書式 (source/eval/p2nnue/p2nnue_kh.h): f32 LE、 [out x dims]、 values[feature * out + row]。 同名 + ".json" で形を検査。
   エンジンは round(QA_KH=254 * w) (0 から遠い方へ丸め) を i16 で持つ。 ここでも同じ式で丸めて、 i16 を 1 個でも超えたら止める。
   a = 1.0 は一時ファイルに書いて元とバイト一致を確かめる (器の確認) だけで、 残さない。
usage: py -3 kh_scale.py 1.25 1.5
"""
import filecmp, io, json, math, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SRC = "E:/shogi/p2shogi/tmp_mob/khhr/kh_final.bin"
QA = 254
js = json.load(io.open(SRC + ".json", encoding="utf-8"))
out, dims, kd = js["out"], js["dims"], js["k_dims"]
raw = open(SRC, "rb").read()
assert len(raw) == out * dims * 4, (len(raw), out, dims)
v = np.frombuffer(raw, dtype="<f4").astype(np.float64)
assert np.all(np.isfinite(v))


def q16(w):
    x = QA * w
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5))


def make(a, path):
    w = (v * a).astype("<f4")
    q = q16(w.astype(np.float64))
    over = int(np.sum((q > 32767) | (q < -32768)))
    M = q.reshape(dims, out)                       # [feature][row]
    hpart = np.abs(M[kd:]).sum(axis=0)             # 行ごとに H 部の全列の |q| を足した最悪値 (同時に全部立つことはない)
    info = dict(a=a, max_abs_q=int(np.abs(q).max()), nonzero_k=int((M[:kd] != 0).sum()), nonzero_h=int((M[kd:] != 0).sum()),
                over_i16=over, worst_row_sum_abs_h=int(hpart.max()))
    if over:
        raise SystemExit("🚨a=%g: 丸めた後に i16 を超える要素が %d 個。 止める" % (a, over))
    with open(path, "wb") as f:
        f.write(w.tobytes())
    js2 = dict(js); js2["kh_scale_a"] = a; js2["kh_scale_src"] = SRC
    with io.open(path + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(js2, f, indent=2, sort_keys=True, ensure_ascii=False)
    return info


# 器の確認: a = 1.0 でバイト一致
tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kh_x100_check.bin")
info = make(1.0, tmp)
same = filecmp.cmp(tmp, SRC, shallow=False)
print("a=1.0: 元とバイト一致 %s / %s" % (same, info))
os.remove(tmp); os.remove(tmp + ".json")
if not same:
    raise SystemExit("🚨a=1.0 で元とバイト一致しない。 止める")
for s in sys.argv[1:]:
    a = float(s)
    dst = "E:/shogi/p2shogi/tmp_mob/khhr/kh_x%03d.bin" % int(round(a * 100))
    if os.path.exists(dst):
        raise SystemExit("🚨既に在る (上書きしない): " + dst)
    info = make(a, dst)
    # 書いたものを読み戻して a 倍になっているか
    back = np.frombuffer(open(dst, "rb").read(), dtype="<f4").astype(np.float64)
    nz = v != 0
    ratio = back[nz] / v[nz]
    print("a=%g -> %s / %s / 読み戻しの比 min %.6f max %.6f" % (a, dst, info, ratio.min(), ratio.max()))
