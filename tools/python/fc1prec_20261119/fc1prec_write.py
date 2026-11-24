# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""FC1PREC net の書き出し器 (2026-11-19 / PREREG-20261119-FC1PREC.md / エンジン側は p2nnue.cpp・p2nnue_opn.h の P2_FC1_PREC)。
fti32_write.py (runs/fti32_20261118) の写しから作った。

  py -3 fc1prec_write.py [--opn] <元 FTI32 net (版 0x7AF32F21)> <weights.bin> <出力 .bin> [--ref <基準の側ファイル>]

元の net のバイト列は FT・PSQT・threat・fc0 をそのまま写し、 末尾の bucket (head) 8 個のうち次の 3 つだけを weights.bin の f32 から作り直す:
  fc1 の重み  = clip(round(128 · l2w), -127, 127)      (i8 のまま。 刻み 64 -> 128。 切られる数を数え、 1 つでもあれば止める)
  fc1 の bias = round(127·4·128 · l2b) = round(65024 · l2b)   (fc0 活性が 2 ビット細かい分 x4、 重みの刻みで x2)
  fc2 の bias = round(127·4·64 · l3b) = round(32512 · l3b)    (fc1 活性が 2 ビット細かい分 x4。 fc2 の重みは刻み 64 のまま変えない)
round は学習器 (Rust) と同じ「0.5 は 0 から遠い側」。 書き換える前に、 元の値が同じ式の刻み 64 版 (bias 8128) と 1 つ残らず一致するかを確かめる
(一致しなければ区画の位置か net の取り違えなので止める)。
arch の fc1 の token "AffineTransform[L3<-30]" を "AffineTransformF1P128[L3<-30]" にし (旧 exe は読み込みで止まる)、 版を 0x7AF32F22 にする。
--ref: 本体だけ。 エンジンの計測 (P2_FC1_REF、 段 3 (d)) が読む f64 の後段の重み (fc0 は f32 の merged、 L1Q128 の目盛り) を書く。
出力が既にあれば止める (上書きしない)。
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k256_20261117"))
from k256_merge import read_weight  # noqa: E402

VER_FTI32, VER_FC1P = 0x7AF32F21, 0x7AF32F22
NB, L1, L2IN = 8, 16, 30
M = L1 - 1


def rnd(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def pad32(n):
    return (n + 31) // 32 * 32


def main():
    a = sys.argv[1:]
    opn = a[0] == "--opn"
    if opn:
        a = a[1:]
    ref = None
    if "--ref" in a:
        i = a.index("--ref"); ref = a[i + 1]; del a[i:i + 2]
        assert not opn, "--ref は本体だけ"
    src, wpath, outp = a
    for p in [outp] + ([ref] if ref else []):
        if os.path.exists(p):
            raise SystemExit("出力が既にある (上書きしない): " + p)
    FIN, L3 = (2048, 32) if opn else (1024, 64)   # fc0 の入力幅 (pairwise 後)、 fc1 の出力
    b = open(src, "rb").read()
    ver, nh, alen = struct.unpack_from("<III", b, 0)
    arch = b[12:12 + alen].decode("ascii")
    if ver != VER_FTI32:
        raise SystemExit(f"版 0x{ver:08X} は 0x7AF32F21 (FTI32) でない")
    tok = f"AffineTransform[{L3}<-{L2IN}]"
    assert arch.count(tok) == 1 and "F1P128" not in arch, arch
    tp = arch.index(tok)
    assert arch[:tp].count("AffineTransform[") == 1, "fc1 が 2 番目の AffineTransform でない: " + arch
    arch2 = arch[:tp] + "AffineTransformF1P128[" + arch[tp + len("AffineTransform["):]
    bs = 4 + 4 * L1 + L1 * pad32(FIN) + 4 * L3 + L3 * pad32(L2IN) + 4 + pad32(L3)
    start = len(b) - NB * bs
    l2w = read_weight(wpath, "l2w").astype(np.float64).reshape(2 * M, NB, L3)
    l2b = read_weight(wpath, "l2b").astype(np.float64)
    l3b = read_weight(wpath, "l3b").astype(np.float64)
    body = bytearray(b[12 + alen:])
    off0 = start - (12 + alen)
    nbad, ncut, mx = 0, 0, 0
    for k in range(NB):
        p = off0 + k * bs + 4 + 4 * L1 + L1 * pad32(FIN)
        pw = p + 4 * L3
        p2 = pw + L3 * pad32(L2IN)
        ob1 = np.frombuffer(bytes(body[p:pw]), "<i4")
        ow1 = np.frombuffer(bytes(body[pw:p2]), "i1").reshape(L3, pad32(L2IN))
        ob2 = struct.unpack_from("<i", body, p2)[0]
        wk = l2w[:, k, :].T
        nbad += int((ob1 != rnd(8128 * l2b[k * L3:(k + 1) * L3])).sum()) + int((ow1[:, :L2IN] != np.clip(rnd(64 * wk), -128, 127)).sum())
        nbad += int(ow1[:, L2IN:].any()) + int(ob2 != rnd(8128 * l3b[k]))
        q = rnd(128 * wk)
        ncut += int((np.abs(q) > 127).sum()); mx = max(mx, int(np.abs(q).max()))
        nb1 = rnd(65024 * l2b[k * L3:(k + 1) * L3]); nb2 = rnd(32512 * l3b[k])
        assert np.abs(nb1).max() < 2 ** 31 and abs(nb2) < 2 ** 31
        nw1 = np.zeros((L3, pad32(L2IN)), np.int8); nw1[:, :L2IN] = np.clip(q, -127, 127)
        body[p:pw] = nb1.astype("<i4").tobytes()
        body[pw:p2] = nw1.tobytes()
        struct.pack_into("<i", body, p2, int(nb2))
    if nbad:
        raise SystemExit(f"元の fc1 / fc2 bias が刻み 64 の式と {nbad} 個違う (区画の位置か net の取り違え)。 書かない")
    if ncut:
        raise SystemExit(f"fc1 の重みが刻み 128 で {ncut} 個 i8 に入らない (max|q| {mx})。 書かない")
    a2 = arch2.encode("ascii")
    out = struct.pack("<III", VER_FC1P, nh, len(a2)) + a2 + bytes(body)
    assert len(out) == len(b) + len(a2) - alen
    open(outp, "wb").write(out)
    print(f"書き出し {outp}  {'OPN' if opn else '本体'}  fc1 重み 刻み 128 max|q| {mx} (切られ 0)  元の値の照合 不一致 0\n"
          f"  大きさ {len(b)} -> {len(out)} byte  版 0x{VER_FC1P:08X}\n  arch {arch2}")
    if ref:
        l1q = "AffineTransformL1Q128[" in arch
        T = {n: read_weight(wpath, n).astype(np.float64) for n in ("l1w", "l1fw", "l1b", "l1fb", "l3w")}
        merged = T["l1w"].reshape(FIN, NB, L1) + T["l1fw"].reshape(FIN, L1)[:, None, :]
        biasf = T["l1b"].reshape(NB, L1) + T["l1fb"][None, :]
        SC = np.r_[np.full(M, 128.0 if l1q else 64.0), 64.0]; BS = np.r_[np.full(M, 16256.0 if l1q else 8128.0), 8128.0]
        l3w = T["l3w"].reshape(L3, NB)
        parts = [b"P2FC1REF", struct.pack("<5i", NB, L1, FIN, L3, L2IN)]
        for k in range(NB):
            arrs = [BS * biasf[k], SC[:, None] * merged[:, k, :].T, 127.0 * l2b[k * L3:(k + 1) * L3], l2w[:, k, :].T,
                    np.array([8128.0 * l3b[k]]), 64.0 * l3w[:, k]]
            parts += [np.ascontiguousarray(x, dtype="<f8").tobytes() for x in arrs]
        open(ref, "wb").write(b"".join(parts))
        print(f"基準の側ファイル {ref}  (fc0 の目盛り {'L1Q128' if l1q else '64'})")


if __name__ == "__main__":
    main()
