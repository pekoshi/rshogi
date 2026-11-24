# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""FTI32 net の書き出し器 (2026-11-18 / PREREG-20261118-FTI32.md / エンジン側は p2nnue.cpp の P2_FT_I32)。

  py -3 fti32_write.py <元 quantised.bin> <weights.bin> <k> <出力 .bin>
  py -3 fti32_write.py --opn ...   (OPN の net。 FT は K16 + 因子行、 2026-11-18 追加)
  py -3 fti32_write.py --check <元 quantised.bin> <weights.bin>     (k=1 の量子化が元の net の FT と一致するかだけ見る)

元の net (刻み 254 の i16 FT) の FT の LEB128 ブロックだけを、 weights.bin の l0w / l0b から作った k 倍刻みの値
  重み = round(float32(254·k) · l0w)   (float32 の積を numpy の偶数丸め。 ftdiag_s0_fti32.py の FTK と同じ式)
  bias = round(254·k · float64(l0b))
に差し替える。 bias と重みは別の 2 ブロック (エンジンの loader は 1 ブロック / 2 ブロックの両方を読む)。
arch の FT 幅の前に "I32K<k>:" を入れ ("[73305->I32K8:1024x2]")、 版番号を 0x7AF32F21 にする。 PSQT 以降は 1 byte も変えない。
i16 に入らない値があれば止まる。 出力が既にあれば止まる (上書きしない)。
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k256_20261117"))
from k256_merge import read_weight  # noqa: E402

IN, FT = 73305, 1024
MAGIC = b"COMPRESSED_LEB128"
VER_FTI32 = 0x7AF32F21


OPN = False   # --opn: OPN の net (K16 26064 行 x 幅 2048、 l0w の先頭 1629 行 = 因子行を各区分に足す。 s0_opn.py と同じ式)


def quant_ft(wpath, k):
    w = read_weight(wpath, "l0w")
    b = read_weight(wpath, "l0b").astype(np.float64)
    if OPN:
        PI = 1629
        l0 = w.reshape(-1, 2048)
        ftw32 = (l0[PI:].reshape(16, PI, 2048) + l0[:PI][None, :, :]).reshape(16 * PI, 2048)
        return np.round(np.float32(254.0 * k) * ftw32), np.round(254.0 * k * b)
    qw = np.round(np.float32(254.0 * k) * w.reshape(IN, FT))
    qb = np.round(254.0 * k * b)
    return qw, qb


def leb_decode(buf):
    """signed LEB128 の列 (numpy 版、 1〜3 byte)。"""
    a = np.frombuffer(buf, dtype=np.uint8)
    end = np.flatnonzero((a & 0x80) == 0)
    start = np.r_[0, end[:-1] + 1]
    n = end - start + 1
    assert n.max() <= 3, n.max()
    v = np.zeros(len(end), np.int64)
    for i in range(3):
        m = n > i
        v[m] |= (a[start[m] + i].astype(np.int64) & 0x7F) << (7 * i)
    last = a[end].astype(np.int64)
    neg = (last & 0x40) != 0
    v[neg] -= (1 << (7 * n[neg]))
    return v


def leb_encode(v):
    v = np.asarray(v, dtype=np.int64)
    n = np.where((v >= -64) & (v < 64), 1, np.where((v >= -8192) & (v < 8192), 2, 3))
    assert ((v >= -(1 << 20)) & (v < (1 << 20))).all()
    out_len = n.sum()
    out = np.zeros(out_len, np.uint8)
    pos = np.r_[0, np.cumsum(n)[:-1]]
    for i in range(3):
        m = n > i
        byte = (v[m] >> (7 * i)) & 0x7F
        cont = n[m] > i + 1
        out[pos[m] + i] = (byte | np.where(cont, 0x80, 0)).astype(np.uint8)
    assert (leb_decode(out.tobytes()) == v).all()
    return out.tobytes()


def parse(b):
    ver, nh, alen = struct.unpack_from("<III", b, 0)
    arch = b[12:12 + alen].decode("ascii")
    p = 12 + alen + 4  # ft_hash
    blocks = []
    while b[p:p + 17] == MAGIC:
        comp = struct.unpack_from("<I", b, p + 17)[0]
        blocks.append((p, p + 21, p + 21 + comp))
        p += 21 + comp
    assert len(blocks) in (1, 2), len(blocks)
    return ver, nh, arch, blocks, p


def main():
    global OPN
    if sys.argv[1] == "--opn":
        OPN = True
        del sys.argv[1]
    if sys.argv[1] == "--check":
        src, wpath = sys.argv[2:4]
        b = open(src, "rb").read()
        ver, nh, arch, blocks, rest = parse(b)
        vals = np.concatenate([leb_decode(b[s:e]) for _, s, e in blocks])
        qw, qb = quant_ft(wpath, 1)
        ref = np.r_[qb, qw.ravel()]
        assert vals.size == ref.size, (vals.size, ref.size)
        d = vals != ref
        print(f"arch {arch}\nversion 0x{ver:08X} / blocks {len(blocks)} / FT 値 {vals.size} / k=1 の量子化と不一致 {int(d.sum())}"
              f" (bias {int(d[:FT].sum())})")
        return
    src, wpath, k, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    assert k in (2, 4, 8)
    if os.path.exists(outp):
        raise SystemExit("出力が既にある (上書きしない): " + outp)
    b = open(src, "rb").read()
    ver, nh, arch, blocks, rest = parse(b)
    assert "->I32K" not in arch, "元の net が既に FTI32 形式"
    tok = "[26064->2048x2]" if OPN else "[73305->1024x2]"
    assert arch.count(tok) == 1, arch
    qw, qb = quant_ft(wpath, k)
    for name, a in (("重み", qw), ("bias", qb)):
        if np.abs(a).max() > 32767:
            raise SystemExit(f"{name} が i16 に入らない: max|q| {np.abs(a).max()}")
    arch2 = arch.replace(tok, tok.replace("->", f"->I32K{k}:"))
    a2 = arch2.encode("ascii")
    out = bytearray()
    out += struct.pack("<III", VER_FTI32, nh, len(a2)) + a2
    out += b[12 + len(arch.encode("ascii")):12 + len(arch.encode("ascii")) + 4]  # ft_hash
    for arr in (qb, qw.ravel()):
        e = leb_encode(arr.astype(np.int64))
        out += MAGIC + struct.pack("<I", len(e)) + e
    out += b[rest:]
    open(outp, "wb").write(bytes(out))
    print(f"書き出し {outp}  k={k}  重み max|q| {int(np.abs(qw).max())} / bias max|q| {int(np.abs(qb).max())}"
          f"  大きさ {len(b)} -> {len(out)} byte\narch {arch2}")


if __name__ == "__main__":
    main()
