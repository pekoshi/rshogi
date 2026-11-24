# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""K256 案 A (DESIGN-20261117-K256-FABLE 4.3): 学習器の追加列 `threat_ext_w` を K' 幅の net に差し込む器。

net の並び (make_knets.py と同じ。 p2nnue.cpp load の読み順):
  version u32 / nethash u32 / arch_len u32 / arch / ft_hash u32 / LEB128 block 1 つか 2 つ
  PSQT = i32 x (NUM_BUCKETS + HALFKA_DIM * NUM_BUCKETS)
  threat 表 = i8 x groups x THREAT_DIM x K0   (群主・特徴主: 1 行 = K0 列)
  残り (layer stack ほか) はそのまま写す

学習器の重み (weights.bin = 「名前 + 改行 + u64 要素数 + f32 x 要素数」の並び):
  threat_ft_w  (K0 x dims*groups) column-major = values[(g*dims + f) * K0 + k]   (旧表 = quantised.bin の表と同じ並び)
  threat_ext_w (N  x dims)        column-major = values[f * N + j]                (1 群、 全群に同じ値を書く)

量子化 (shogi_layerstack.rs の threat 書き出しと同じ): q = round(254 * w) (0.5 は 0 から遠い側へ) を [-128, 127] に clamp。

K' = K0 + N 幅の 1 行の並び (エンジンの THREAT_LO_DIM = K'/2: 列 c < K'/2 -> FT[c]、 c >= K'/2 -> FT[512 + c - K'/2]):
  [旧 0..32) | 追加 0..N/2) | 旧 32..64) | 追加 N/2..N)
  -> 追加の前半は FT[32, 32+N/2)、 後半は FT[544, 544+N/2) = 学習器の kh_add_rows(off = K0/2) と同じ行。

使い方:
  py -3 k256_merge.py control <weights.bin> <quantised.bin>
      陽性対照: weights.bin の threat_ft_w を同じ規則で量子化し、 quantised.bin の threat 表とバイト一致を確かめる。
      (偶数丸め np.round に替えたときに不一致が出ることも数えて、 比較が丸めの違いを検出できることを示す)
  py -3 k256_merge.py merge <net.bin> <weights.bin | --zero> <K'> <out.bin>
      net.bin (K0=64 の現職の形) の threat 表を K' 幅に広げ、 追加列に threat_ext_w を量子化して置く。
      --zero なら追加列を 0 にする (= make_knets.py widen と同じ出力になるはず = 恒等性の検算)。
  py -3 k256_merge.py samebody <a.bin> <b.bin>
      2 つの K0=64 の net が threat 表も含めてバイト一致するかを区切りごとに出す (凍結の検算)。
  py -3 k256_merge.py thrsplice <net.bin> <weights.bin> <out.bin>
      U1a (DESIGN-20261117-T64USE-FABLE 3.2): net.bin (K0=64) の threat 表だけを weights.bin の threat_ft_w の量子化で差し替える
      (fc0・KH・OPN の目盛りは 1 byte も変わらない)。 差し替え前後で変わった要素数と相対変化 |Δw|/|w| を出す。
  py -3 k256_merge.py extstat <weights.bin>
      threat_ext_w の大きさ・量子化後の非 0 率・飽和件数を出す (学習の進み具合の確認)。
"""
import os
import re
import struct
import sys

import numpy as np

HALFKA_DIM = 73305
NUM_BUCKETS = 8
K0 = 64
QA_FT = 254.0
MAGIC = b"COMPRESSED_LEB128"


def parse(b):
    ver, nh, alen = struct.unpack_from("<III", b, 0)
    arch = b[12:12 + alen].decode("ascii")
    p = 12 + alen
    p += 4  # ft_hash
    nblk = 0
    while b[p:p + 17] == MAGIC:
        comp = struct.unpack_from("<I", b, p + 17)[0]
        p += 17 + 4 + comp
        nblk += 1
    assert nblk in (1, 2), nblk
    assert "PSQT=" in arch
    p_psqt = p
    p += 4 * (NUM_BUCKETS + HALFKA_DIM * NUM_BUCKETS)
    assert "ThreatProfile=" not in arch  # profile 0 = prefix 無し
    assert "ThreatI16=1" not in arch     # i8 表
    tdim = int(re.search(r"Threat=(\d+)", arch).group(1))
    m = re.search(r"ThreatGroups=(\d+):", arch)
    groups = int(m.group(1)) if m else 1
    p_thr = p
    p_rest = p_thr + tdim * K0 * groups
    assert p_rest < len(b)
    return dict(arch=arch, hdr_end=12 + alen, p_psqt=p_psqt, p_thr=p_thr, p_rest=p_rest,
                tdim=tdim, groups=groups, nblk=nblk)


def read_weight(path, name):
    """weights.bin から name の f32 配列を返す (無ければ None)。 全体を読まずに見出しだけ辿る。"""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        off = 0
        while off < size:
            f.seek(off)
            head = f.read(512)
            nl = head.index(b"\n")
            nm = head[:nl].decode("utf-8")
            n = struct.unpack_from("<Q", head, nl + 1)[0]
            data_off = off + nl + 1 + 8
            if nm == name:
                return np.fromfile(path, dtype="<f4", count=n, offset=data_off)
            off = data_off + 4 * n
    return None


def quantise(v):
    """Rust の `(254.0 * v as f64).round().clamp(-128, 127) as i8` と同じ (round = 0.5 は 0 から遠い側)。"""
    x = QA_FT * v.astype(np.float64)
    r = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))
    return np.clip(r, -128, 127).astype(np.int8)


def cmd_control(wpath, qpath):
    b = open(qpath, "rb").read()
    d = parse(b)
    thr = np.frombuffer(b[d["p_thr"]:d["p_rest"]], dtype=np.int8)
    w = read_weight(wpath, "threat_ft_w")
    assert w is not None, "threat_ft_w が weights.bin に無い"
    assert w.size == thr.size, (w.size, thr.size)
    q = quantise(w)
    bad = int(np.count_nonzero(q != thr))
    x = QA_FT * w.astype(np.float64)
    bad_even = int(np.count_nonzero(np.clip(np.round(x), -128, 127).astype(np.int8) != thr))
    bad_trunc = int(np.count_nonzero(np.clip(np.trunc(x), -128, 127).astype(np.int8) != thr))
    sat = int(np.count_nonzero(np.abs(x) > 127.5))
    print("arch=%s" % d["arch"])
    print("threat 表 %d 要素 (dims %d x K0 %d x 群 %d) / 非 0 %.2f%% / 飽和 %d" % (
        thr.size, d["tdim"], K0, d["groups"], 100.0 * np.count_nonzero(thr) / thr.size, sat))
    print("CONTROL_MISMATCH        %d   (本器の規則。 0 なら合格)" % bad)
    print("  参考: 偶数丸め (np.round) %d / 切り捨て %d  (0 でなければ、 比較が丸め規則の違いを検出できる)" % (bad_even, bad_trunc))
    print("CONTROL %s" % ("PASS" if bad == 0 and (bad_even > 0 or bad_trunc > 0) else "FAIL"))
    return 0 if bad == 0 else 1


def widen(old_rows, ext_rows_q, K):
    """old_rows: int8 (rows, 64)。 ext_rows_q: int8 (dims, N) か None (= 0)。 返り値 int8 (rows, K)。"""
    n = K - K0
    rows = old_rows.shape[0]
    out = np.zeros((rows, K), dtype=np.int8)
    h = K // 2
    out[:, :32] = old_rows[:, :32]
    out[:, h:h + 32] = old_rows[:, 32:]
    if ext_rows_q is not None:
        dims = ext_rows_q.shape[0]
        assert rows % dims == 0
        g = rows // dims
        e = np.tile(ext_rows_q, (g, 1))  # 全群に同じ値
        out[:, 32:h] = e[:, :n // 2]
        out[:, h + 32:] = e[:, n // 2:]
    return out


def cmd_merge(npath, wpath, K, opath):
    assert K > K0 and (K - K0) % 32 == 0, K
    n = K - K0
    b = open(npath, "rb").read()
    d = parse(b)
    old = np.frombuffer(b[d["p_thr"]:d["p_rest"]], dtype=np.int8).reshape(-1, K0)
    if wpath == "--zero":
        ext_q = None
        print("追加列 = 0 (--zero)")
    else:
        e = read_weight(wpath, "threat_ext_w")
        assert e is not None, "threat_ext_w が weights.bin に無い (--threat-ext-out の run の checkpoint か?)"
        assert e.size == n * d["tdim"], "threat_ext_w %d 要素 != N %d x dims %d (K' を取り違えていないか)" % (e.size, n, d["tdim"])
        ext_q = quantise(e).reshape(d["tdim"], n)
        x = QA_FT * e.astype(np.float64)
        print("threat_ext_w: N %d x dims %d / mean|w| %.6f / max|w| %.6f / 量子化後の非 0 %.3f%% / 飽和 %d" % (
            n, d["tdim"], float(np.abs(e).mean()), float(np.abs(e).max()),
            100.0 * np.count_nonzero(ext_q) / ext_q.size, int(np.count_nonzero(np.abs(x) > 127.5))))
    wide = widen(old, ext_q, K)
    with open(opath, "wb") as f:
        f.write(b[:d["p_thr"]])  # 見出し・FT・PSQT (arch は K に依存しない = K は build 側の定数)
        f.write(wide.tobytes())
        f.write(b[d["p_rest"]:])
    print("wrote %s (%d bytes) K'=%d groups=%d" % (opath, os.path.getsize(opath), K, d["groups"]))
    return 0


def cmd_samebody(a, b_):
    A = open(a, "rb").read()
    B = open(b_, "rb").read()
    da, db = parse(A), parse(B)
    ok = True
    for name, s, e in [("見出し+FT", 0, "p_psqt"), ("PSQT", "p_psqt", "p_thr"), ("threat", "p_thr", "p_rest"), ("残り", "p_rest", None)]:
        sa = A[da[s] if isinstance(s, str) else s:(da[e] if e else len(A))]
        sb = B[db[s] if isinstance(s, str) else s:(db[e] if e else len(B))]
        same = sa == sb
        ok &= same
        print("  %-10s %s (%d / %d bytes)" % (name, "一致" if same else "★不一致", len(sa), len(sb)))
    print("SAMEBODY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_thrsplice(npath, wpath, opath):
    b = open(npath, "rb").read()
    d = parse(b)
    old = np.frombuffer(b[d["p_thr"]:d["p_rest"]], dtype=np.int8)
    w = read_weight(wpath, "threat_ft_w")
    assert w is not None and w.size == old.size, "threat_ft_w の要素数が net の表と合わない"
    q = quantise(w)
    ch = int(np.count_nonzero(q != old))
    o32 = old.astype(np.float64)
    rel = float(np.linalg.norm(q.astype(np.float64) - o32) / max(np.linalg.norm(o32), 1e-12))
    per = old.size // d["groups"]
    grp = ["%d" % int(np.count_nonzero(q[g * per:(g + 1) * per] != old[g * per:(g + 1) * per])) for g in range(d["groups"])]
    print("threat_ft_w -> i8: 変わった要素 %d / %d (群ごと %s) / 相対変化 |Δq|/|q| %.5f / 飽和 %d" % (
        ch, old.size, ",".join(grp), rel, int(np.count_nonzero(np.abs(QA_FT * w.astype(np.float64)) > 127.5))))
    with open(opath, "wb") as f:
        f.write(b[:d["p_thr"]])
        f.write(q.tobytes())
        f.write(b[d["p_rest"]:])
    print("wrote %s (%d bytes)" % (opath, os.path.getsize(opath)))
    return 0


def cmd_extstat(wpath):
    e = read_weight(wpath, "threat_ext_w")
    if e is None:
        print("threat_ext_w 無し")
        return 1
    q = quantise(e)
    x = QA_FT * e.astype(np.float64)
    print("threat_ext_w %d 要素 / mean|w| %.6f / max|w| %.6f / 量子化後の非 0 %.3f%% / mean|q| %.4f / 飽和 %d" % (
        e.size, float(np.abs(e).mean()), float(np.abs(e).max()), 100.0 * np.count_nonzero(q) / q.size,
        float(np.abs(q.astype(np.int32)).mean()), int(np.count_nonzero(np.abs(x) > 127.5))))
    return 0


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    m = sys.argv[1]
    if m == "control" and len(sys.argv) == 4:
        return cmd_control(sys.argv[2], sys.argv[3])
    if m == "merge" and len(sys.argv) == 6:
        return cmd_merge(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    if m == "samebody" and len(sys.argv) == 4:
        return cmd_samebody(sys.argv[2], sys.argv[3])
    if m == "thrsplice" and len(sys.argv) == 5:
        return cmd_thrsplice(sys.argv[2], sys.argv[3], sys.argv[4])
    if m == "extstat" and len(sys.argv) == 3:
        return cmd_extstat(sys.argv[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
