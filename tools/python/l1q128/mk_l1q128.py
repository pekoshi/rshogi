# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""L1Q128 段階 1 の net を書き出す (学習器を通さない)。 2026-11-16 (設計 DESIGN-20261116-L1Q128-FABLE.md 3.3)。

  quantised.bin の LayerStack (8 bucket) のうち **fc0 の本体 15 行 (行 0..14)** の bias と重みだけを
  raw.bin (f32) から 128 目盛りで作り直して差し替える。 skip 行 (行 15)・fc1・fc2・FT・その他の全バイトは写すだけ。
    本体 bias  : round(16256 * (l1b[bk] + l1fb))   (i32、 127*128)
    本体 重み  : clip(round(128 * (l1w[bk] + l1fw)), -128, 127)   (i8、 13 本前後が飽和する = 設計どおり飽和させる)
  arch の 3 番目の token `AffineTransform[16<-2048]` を `AffineTransformL1Q128[16<-2048]` に書き換える (唯一の目印)。
  旧 exe はこの token を数えられず「arch dims parse failed」で必ず止まる。

  自己検算 (必須・失敗したら書かない): raw.bin から**現行の式** (bias x8128 / 重み x64) で作り直した
  16 行 x 8 bucket の bias と重みが、 元ファイルと**全バイト一致**すること (raw.bin と quantised.bin の組・行の対応の検査)。

  丸めは学習器 (shogi_layerstack.rs) と同じ規則: f32 どうしで足してから f64 に上げ、 0 から遠い方へ丸める
  (Rust の f64::round)。 np.round (偶数丸め) との食い違い (x.5 ちょうど) の本数も表示する。

usage: py -3 mk_l1q128.py <元 quantised.bin> <raw.bin> <出力.bin | ->
   出力に `-` を渡すと自己検算と表示だけ行い、 何も書かない (元ファイルは header と末尾の LayerStack だけ読む)。
"""
import ctypes, hashlib, io, os, struct, sys
import numpy as np

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

NB, L1, M, FC0_IN = 8, 16, 15, 1024       # bucket 数 / fc0 出力 (本体 15 + skip 1) / fc0 入力 (pairwise 後 512 x 2 視点)
L2_IN_PAD, L3 = 32, 64
BUCKET_LEN = 4 + (4 * L1 + L1 * FC0_IN) + (4 * L3 + L3 * L2_IN_PAD) + (4 + L3)   # = 18,824 B
OLD_TOK, NEW_TOK = "AffineTransform[", "AffineTransformL1Q128["
FC0_TOK = "AffineTransform[16<-2048]"
NEED = ("l1w", "l1b", "l1fw", "l1fb")
# 設計書 2.1 の一覧 (a-4 の飽和 13 本)。 違う net では一致しなくてよいが、 本数と位置は必ず表示する。
SAT_A4 = [(0, 2, 659, -135), (1, 5, 33, 138), (1, 5, 769, 148), (1, 5, 909, 149), (1, 12, 659, 140), (2, 5, 909, 131),
          (2, 8, 769, 134), (3, 1, 33, -150), (3, 12, 659, 142), (3, 13, 769, 141), (4, 13, 856, 132), (5, 8, 748, 145),
          (7, 8, 856, 136)]


def die(msg, code=2):
    print("🚨" + msg)
    sys.exit(code)


def read_raw_tensors(path, names):
    """BULLETRW から必要な tensor だけを seek で読む (525 MB 全体は読まない)。"""
    out = {}
    with io.open(path, "rb") as f:
        if f.read(8) != b"BULLETRW":
            die("raw.bin が BULLETRW ではない: " + path)
        _ver, cnt = struct.unpack("<II", f.read(8))
        for _ in range(cnt):
            ln, = struct.unpack("<I", f.read(4))
            nm = f.read(ln).decode("ascii")
            ne, = struct.unpack("<I", f.read(4))
            if nm in names:
                out[nm] = np.frombuffer(f.read(ne * 4), dtype="<f4").copy()
            else:
                f.seek(ne * 4, 1)
    miss = [n for n in names if n not in out]
    if miss:
        die("raw.bin に tensor が無い: %s" % miss)
    return out


def round_away(x):
    """0 から遠い方へ丸める (Rust の f64::round と同じ)。"""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def build(T, bk, qw, qb):
    """bucket bk の fc0 (16 行) を目盛り (重み qw / bias qb) で量子化した値 (i64 の bias[16], 丸めた重み[16][1024] (飽和前))。
       添字は学習器と同じ: l1w は列優先 [in][NB*16]、 l1fw は [in][16]。 f32 で足してから f64 に上げる。"""
    l1w = T["l1w"].reshape(FC0_IN, NB * L1)[:, bk * L1:(bk + 1) * L1]   # [1024][16] f32
    l1fw = T["l1fw"].reshape(FC0_IN, L1)
    w = (l1w + l1fw).T.astype(np.float64)                               # f32 で足す -> f64
    b = (T["l1b"][bk * L1:(bk + 1) * L1] + T["l1fb"]).astype(np.float64)
    wr = round_away(qw * w)
    br = round_away(qb * b)
    ties = int((wr != np.round(qw * w)).sum()) + int((br != np.round(qb * b)).sum())
    return br.astype(np.int64), wr.astype(np.int64), ties


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    src, rawp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    dry = outp == "-"
    if not dry and os.path.exists(outp):
        die("出力が既にある (上書きしない): " + outp, 8)
    size = os.path.getsize(src)
    with io.open(src, "rb") as f:
        head = f.read(12)
        _ver, _hash, alen = struct.unpack("<III", head)
        arch = f.read(alen).decode("ascii")
        ls_off = size - NB * BUCKET_LEN
        f.seek(ls_off)
        ls = bytearray(f.read(NB * BUCKET_LEN))
    # ---- 1. ヘッダの関門
    if NEW_TOK in arch:
        die("既に L1Q128 の net (arch に %s がある)" % NEW_TOK)
    if "HandCountDense=" in arch:
        die("HandCountDense= 付きの net は扱わない (fc0 の行が 1056 B)")
    if "[73305->1024x2]" not in arch:
        die("arch が [73305->1024x2] でない: " + arch)
    if arch.count(FC0_TOK) != 1:
        die("arch に %s がちょうど 1 個でない (%d 個): %s" % (FC0_TOK, arch.count(FC0_TOK), arch))
    if arch.count(OLD_TOK) != 3:
        die("AffineTransform[ が 3 個でない (L4 等は対象外): " + arch)
    print("arch: " + arch)
    # ---- 2. LayerStack の位置 = ファイル末尾の 8 bucket。 fc_hash が 18,824 B おきに 8 回並ぶことを確かめる
    h0 = bytes(ls[0:4])
    if not all(bytes(ls[k * BUCKET_LEN:k * BUCKET_LEN + 4]) == h0 for k in range(NB)):
        die("末尾の 8 bucket に fc_hash が並ばない (bucket 長 %d)" % BUCKET_LEN)
    print("LayerStack: offset %s / bucket %s B x %d / fc_hash %s" % (format(ls_off, ","), format(BUCKET_LEN, ","), NB, h0.hex()))
    T = read_raw_tensors(rawp, NEED)
    for n in NEED:
        print("  raw %-5s %d 要素" % (n, T[n].size))
    if T["l1w"].size != FC0_IN * NB * L1 or T["l1fw"].size != FC0_IN * L1 or T["l1b"].size != NB * L1 or T["l1fb"].size != L1:
        die("raw.bin の L1 tensor の大きさが想定と違う")
    # ---- 3. 自己検算: 現行の式で 16 行 x 8 bucket を作り直して全バイト一致
    ng = 0; ties64 = 0
    for bk in range(NB):
        base = bk * BUCKET_LEN + 4
        b64, w64, t = build(T, bk, 64.0, 8128.0); ties64 += t
        bias_b = b64.astype("<i4").tobytes()
        w_b = np.clip(w64, -128, 127).astype(np.int8).tobytes()
        ob = bytes(ls[base:base + 4 * L1]); ow = bytes(ls[base + 4 * L1:base + 4 * L1 + L1 * FC0_IN])
        nb = sum(x != y for x, y in zip(bias_b, ob)); nw = sum(x != y for x, y in zip(w_b, ow))
        if nb or nw:
            print("  bucket %d: 自己検算 不一致 bias %d B / 重み %d B" % (bk, nb, nw))
        ng += nb + nw
    if ng:
        die("自己検算 失敗 (不一致 %d B)。 raw.bin と quantised.bin の組か行の対応が違う。 書かない" % ng, 3)
    print("自己検算 OK: x64 の作り直し 16 行 x 8 bucket (%s B) が元と全バイト一致 (x.5 の丸めの食い違い %d)" % (
        format(NB * (4 * L1 + L1 * FC0_IN), ","), ties64))
    # ---- 4. 本体 15 行を 128 目盛りで書き換える
    sat = []; ties128 = 0; changed = 0; bmax = 0; wrms = []
    for bk in range(NB):
        base = bk * BUCKET_LEN + 4
        b128, w128, t = build(T, bk, 128.0, 16256.0); ties128 += t
        for r, c in np.argwhere(np.abs(w128[:M]) > 127):
            sat.append((bk, int(r), int(c), int(w128[r, c])))
        bmax = max(bmax, int(np.abs(b128[:M]).max()))
        if np.abs(b128[:M]).max() >= 2 ** 31:
            die("bias が i32 を超える")
        wrms.append(float(np.sqrt((w128[:M].astype(np.float64) ** 2).mean())))
        new_bias = bytearray(ls[base:base + 4 * L1])
        new_bias[0:4 * M] = b128[:M].astype("<i4").tobytes()
        new_w = bytearray(ls[base + 4 * L1:base + 4 * L1 + L1 * FC0_IN])
        new_w[0:M * FC0_IN] = np.clip(w128[:M], -128, 127).astype(np.int8).tobytes()
        changed += sum(x != y for x, y in zip(new_bias, ls[base:base + 4 * L1]))
        changed += sum(x != y for x, y in zip(new_w, ls[base + 4 * L1:base + 4 * L1 + L1 * FC0_IN]))
        ls[base:base + 4 * L1] = new_bias
        ls[base + 4 * L1:base + 4 * L1 + L1 * FC0_IN] = new_w
    sat.sort()
    print("飽和 (|round(128w)| > 127) %d 本 / 本体 %s 本: %s" % (len(sat), format(NB * M * FC0_IN, ","), sat))
    print("  設計書 2.1 の a-4 の一覧と %s" % ("一致" if sat == sorted(SAT_A4) else "**不一致**"))
    print("  本体 bias |q| 最大 %d / 重み RMS (bucket 別) %s / x.5 の丸めの食い違い %d" % (bmax, [round(x, 2) for x in wrms], ties128))
    print("書き換え対象 %s B (8 x 15 x (4 + 1024))、 うち値が変わったバイト %s" % (format(NB * M * (4 + FC0_IN), ","), format(changed, ",")))
    new_arch = arch.replace(FC0_TOK, "AffineTransformL1Q128[16<-2048]")
    assert new_arch.count(OLD_TOK) == 2 and new_arch.count(NEW_TOK) == 1
    if dry:
        print("(- 指定のため書かない)")
        return 0
    # ---- 5. 書き出し (一時ファイル -> os.replace)
    data = bytearray(io.open(src, "rb").read())
    assert len(data) == size
    if bytes(data[ls_off:ls_off + 4]) != h0:
        die("読み直した元ファイルが変わっている")
    out = bytearray(head[:8]) + struct.pack("<I", len(new_arch)) + new_arch.encode("ascii") \
        + data[12 + alen:ls_off] + ls
    tmp = outp + ".tmp"
    with io.open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, outp)
    sha = lambda bb: hashlib.sha256(bb).hexdigest()
    print("大きさ: 元 %s -> 出力 %s (+%d)" % (format(size, ","), format(len(out), ","), len(out) - size))
    print("sha256 元   %s\nsha256 出力 %s\n-> %s" % (sha(bytes(data)), sha(bytes(out)), outp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
