# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""ngs436/dlsuisho-ponkotsu-distilled から「系譜が学習していない局面」だけを抜く器。

方式:
  1. build-bloom : 系譜の参照 PSV (novel_ext3 / dlsuisho_preext3_residual_v1 ほか) の packed sfen を
                   Bloom filter に入れてファイルに保存する。 偽陽性 = 未学習を捨てる側なので安全。
  2. extract     : ngs436 の shard を curl でストリーミングし (shard をディスクに置かない)、
                   Bloom に「無い」局面だけを書き出す。 shard 001 の先頭 (v8-ponko が読んだ範囲) は規則で捨てる。

鍵は packed sfen 32B だけ (score/move/ply/result/pad は使わない)。 PSV 40B/局面。

使い方の例:
  py -3 -I ponko_novel.py build-bloom --out E:/.../lineage.bloom --bytes 3221225472 --k 3 \
      --ref E:/shogi/p2shogi/data/washiun_dl/novel_ext3.psv \
      --ref E:/shogi/p2shogi/data/washiun_dl/dlsuisho_preext3_residual_v1.psv
  py -3 -I ponko_novel.py extract --bloom E:/.../lineage.bloom --shard 002 --out E:/.../ponko_novel_002.psv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

REC = 40
CHUNK_RECS = 1_600_000  # 64 MB / chunk
URL_FMT = "https://huggingface.co/datasets/ngs436/dlsuisho-ponkotsu-distilled/resolve/main/dlsuisho_unique_{}.bin"
SHARD_BYTES = 19_558_599_240
# v8-ponko (系譜の途中 / v7ft254-l30-53 -> v8-ponko-4) が ponkotsu_WCSC36_001.bin (= ngs436 shard 001 と
# sha256 一致) を 3 ファイル round-robin で 4 SB 読んだ = 先頭 約 1.333 億局面。 256 MB チャンク単位の
# 先読みを見込んで 1.40 億で切る。
SHARD001_SEEN_RECS = 140_000_000
MAGIC = b"PNKBLM01"
HDR = 64

M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
SEED = np.uint64(0x2545F4914F6CDD1D)
C2 = np.uint64(0x9E3779B97F4A7C15)
C3 = np.uint64(0xD6E8FEB86659FD93)


def mix64(x):
    """splitmix64 の finalizer (配列をその場で書き換えない)。"""
    x = x ^ (x >> np.uint64(30))
    x = x * M1
    x = x ^ (x >> np.uint64(27))
    x = x * M2
    return x ^ (x >> np.uint64(31))


def keys_of(raw_u8):
    """(n, 40) uint8 -> packed sfen 32B の 64bit 鍵。"""
    w = np.ascontiguousarray(raw_u8[:, :32]).view("<u8").reshape(-1, 4)
    with np.errstate(over="ignore"):
        h = mix64(w[:, 3] ^ SEED)
        h = mix64(w[:, 2] ^ h)
        h = mix64(w[:, 1] ^ h)
        return mix64(w[:, 0] ^ h)


def bloom_positions(key, nwords, k):
    """鍵ごとに k 個の (語の添字, ビットマスク) を返す。 double hashing。"""
    with np.errstate(over="ignore"):
        h2 = mix64(key ^ C2) | np.uint64(1)
        out = []
        for i in range(k):
            h = key + np.uint64(i) * h2
            word = ((h >> np.uint64(32)) * np.uint64(nwords)) >> np.uint64(32)
            mask = np.uint64(1) << (h & np.uint64(63))
            out.append((word.astype(np.int64), mask))
    return out


def bloom_contains(bits, key, k):
    hit = np.ones(len(key), dtype=bool)
    for word, mask in bloom_positions(key, len(bits), k):
        hit &= (bits[word] & mask) != 0
    return hit


def keep_by_frac(key, frac):
    """鍵から決定的に frac の割合を残す (Bloom の添字と独立な混ぜ方)。"""
    if frac >= 1.0:
        return np.ones(len(key), dtype=bool)
    with np.errstate(over="ignore"):
        r = mix64(key ^ C3) >> np.uint64(40)  # 24 bit
    return r < np.uint64(int(frac * (1 << 24)))


def iter_file_chunks(path, offset=0, limit_bytes=None):
    size = os.path.getsize(path)
    end = size if limit_bytes is None else min(size, offset + limit_bytes)
    end = offset + (end - offset) // REC * REC
    with open(path, "rb") as f:
        f.seek(offset)
        pos = offset
        while pos < end:
            n = min(CHUNK_RECS * REC, end - pos)
            buf = f.read(n)
            if not buf:
                break
            pos += len(buf)
            yield np.frombuffer(buf, dtype=np.uint8).reshape(-1, REC)


# ---------------------------------------------------------------- build-bloom
def cmd_build_bloom(a):
    nwords = a.bytes // 8
    if nwords >= 1 << 32:
        sys.exit("--bytes は 32 GiB 未満にすること (語の添字が 32bit 前提)")
    bits = np.zeros(nwords, dtype=np.uint64)
    t0 = time.time()
    total = 0
    meta_refs = []
    for spec in a.ref:
        # ローカル: path または path@byte_offset@limit_bytes
        # HF     : https://...@byte_start@byte_end (curl で流す。 ディスクに置かない)
        parts = spec.split("@")
        path = parts[0]
        off = int(parts[1]) // REC * REC if len(parts) > 1 else 0
        if path.startswith("http"):
            if len(parts) != 3:
                sys.exit(f"URL の参照は url@start@end で渡すこと: {spec}")
            chunks = iter_curl_chunks(path, off, int(parts[2]) // REC * REC)
        else:
            chunks = iter_file_chunks(path, off, int(parts[2]) if len(parts) > 2 else None)
        n_file = 0
        for raw in chunks:
            key = keys_of(raw)
            for word, mask in bloom_positions(key, nwords, a.k):
                np.bitwise_or.at(bits, word, mask)
            n_file += len(raw)
            total += len(raw)
            if n_file % (CHUNK_RECS * 25) < CHUNK_RECS:
                el = time.time() - t0
                print(f"  {os.path.basename(path)} {n_file / 1e6:.0f}M ({total / el / 1e6:.2f}M rec/s)", flush=True)
        meta_refs.append({"path": path, "offset": off, "records": n_file})
        print(f"[ref] {path} @{off} {n_file} 局面", flush=True)
    fill = float(np.unpackbits(bits[: min(nwords, 1 << 22)].view(np.uint8)).mean())
    meta = {"nwords": int(nwords), "k": a.k, "inserted": int(total), "refs": meta_refs,
            "fill_sample": fill, "est_fpr": fill ** a.k, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = a.out + ".part"
    with open(tmp, "wb") as f:
        hdr = MAGIC + np.array([nwords, a.k], dtype="<u8").tobytes()
        f.write(hdr + b"\0" * (HDR - len(hdr)))
        bits.tofile(f)
    os.replace(tmp, a.out)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[done] {a.out} 挿入 {total} / 充填率 {fill:.3f} / 推定偽陽性 {fill ** a.k:.3%} / {time.time() - t0:.0f}s")


def load_bloom(path):
    with open(path, "rb") as f:
        hdr = f.read(HDR)
    if hdr[:8] != MAGIC:
        sys.exit(f"Bloom ファイルではない: {path}")
    nwords, k = np.frombuffer(hdr[8:24], dtype="<u8")
    bits = np.memmap(path, dtype=np.uint64, mode="r", offset=HDR, shape=(int(nwords),))
    return bits, int(k)


# ---------------------------------------------------------------- extract
def iter_curl_chunks(url, start, end, retries=20):
    """[start, end) を curl で流す。 途中で切れたら現在位置から Range で再開する。"""
    pos = start
    carry = b""
    fails = 0
    while pos < end:
        cmd = ["curl", "-sL", "--fail", "--retry", "5", "-r", f"{pos}-{end - 1}", url]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=-1)  # 11-11: バッファ付き = read(n) が n バイト揃うまで待つ (bufsize=0 だと数十 KB の小片で返り CPU 律速だった)
        try:
            while True:
                buf = p.stdout.read(CHUNK_RECS * REC - len(carry))
                if not buf:
                    break
                buf = carry + buf
                n = len(buf) // REC * REC
                carry = buf[n:]
                if n:
                    pos += n
                    yield np.frombuffer(buf[:n], dtype=np.uint8).reshape(-1, REC)
                    fails = 0
        finally:
            p.stdout.close()
            p.wait()
        if pos + len(carry) >= end:
            break
        fails += 1
        if fails > retries:
            raise RuntimeError(f"curl が {retries} 回続けて失敗: {url} @ {pos}")
        print(f"  [retry {fails}] curl rc={p.returncode} @ {pos}", flush=True)
        carry = b""  # 端数は捨てて境界から取り直す
        time.sleep(min(60, 5 * fails))


def cmd_extract(a):
    bits, k = load_bloom(a.bloom)
    if a.src:
        size = os.path.getsize(a.src)
        start = a.start_rec * REC
        end = min(size, a.end_rec * REC) if a.end_rec else size // REC * REC
        chunks = iter_file_chunks(a.src, start, end - start)
        shard = a.shard or "local"
    else:
        url = URL_FMT.format(a.shard)
        start = a.start_rec * REC
        end = a.end_rec * REC if a.end_rec else SHARD_BYTES
        chunks = iter_curl_chunks(url, start, end)
        shard = a.shard
    seen_head = SHARD001_SEEN_RECS if shard == "001" else 0

    # 陽性対照: Bloom に入れた参照の先頭 1 万局面は全部「有る」と出なければならない
    meta = json.load(open(a.bloom + ".json", encoding="utf-8")) if os.path.exists(a.bloom + ".json") else None
    if meta and not a.skip_control:
        r0 = next(r for r in meta["refs"] if not r["path"].startswith("http"))
        raw = next(iter_file_chunks(r0["path"], r0["offset"], 10_000 * REC))
        miss = int((~bloom_contains(bits, keys_of(raw), k)).sum())
        if miss:
            sys.exit(f"陽性対照に失敗: 参照の先頭 1 万局面のうち {miss} が Bloom に無い (鍵の作り方が不一致)")
        print(f"[control] 参照先頭 1 万局面 すべて Bloom に有り (OK) / 推定偽陽性 {meta['est_fpr']:.2%}")

    tmp = a.out + ".part"
    out = open(tmp, "wb")
    t0 = time.time()
    st = dict(read=0, head_skip=0, seen=0, frac_drop=0, mate=0, written=0)
    rec_index = a.start_rec
    for raw in chunks:
        n = len(raw)
        keep = np.ones(n, dtype=bool)
        if seen_head and rec_index < seen_head:
            m = np.arange(rec_index, rec_index + n) < seen_head
            keep &= ~m
            st["head_skip"] += int(m.sum())
        rec_index += n
        key = keys_of(raw)
        inb = bloom_contains(bits, key, k) & keep
        st["seen"] += int(inb.sum())
        keep &= ~inb
        kf = keep_by_frac(key, a.keep_frac)
        st["frac_drop"] += int((keep & ~kf).sum())
        keep &= kf
        if a.drop_mate:
            sc = raw[:, 32:34].copy().view("<i2").reshape(-1)
            mm = keep & (np.abs(sc.astype(np.int32)) >= 29000)
            st["mate"] += int(mm.sum())
            keep &= ~mm
        sel = raw[keep]
        out.write(sel.tobytes())
        st["read"] += n
        st["written"] += len(sel)
        if st["read"] % (CHUNK_RECS * 10) < CHUNK_RECS:
            el = time.time() - t0
            print(f"  {shard} {st['read'] / 1e6:.0f}M 読 / {st['written'] / 1e6:.1f}M 出 / 既見率 "
                  f"{st['seen'] / max(1, st['read'] - st['head_skip']):.3f} / {st['read'] * REC / el / 1e6:.1f} MB/s",
                  flush=True)
        free_gb = shutil.disk_usage(os.path.dirname(os.path.abspath(a.out))).free / 1e9
        if free_gb < a.min_free_gb:
            print(f"  出力先の空きが {free_gb:.1f} GB (< {a.min_free_gb}) なので打ち切り")
            break
        if a.max_out and st["written"] >= a.max_out:
            print("  --max-out に達したので打ち切り")
            break
    out.close()
    os.replace(tmp, a.out)
    seen_rate = st["seen"] / max(1, st["read"] - st["head_skip"])
    st.update(shard=shard, seen_rate=seen_rate, sec=round(time.time() - t0, 1), out=a.out)
    with open(a.out + ".done", "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    print("[done]", json.dumps(st, ensure_ascii=False))
    # 関門: 既見率が想定 (標本推定 35% + 偽陽性) から大きく外れたら鍵か参照が壊れている
    if st["read"] > 1_000_000 and not (a.seen_lo <= seen_rate <= a.seen_hi):
        print(f"[WARN] 既見率 {seen_rate:.3f} が想定 [{a.seen_lo}, {a.seen_hi}] の外。 参照・Bloom を確認すること")
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-bloom")
    b.add_argument("--ref", action="append", required=True, help="path または path@byte_offset@limit_bytes")
    b.add_argument("--out", required=True)
    b.add_argument("--bytes", type=int, default=3 << 30, help="Bloom の大きさ (既定 3 GiB)")
    b.add_argument("--k", type=int, default=3)
    b.set_defaults(func=cmd_build_bloom)
    e = sub.add_parser("extract")
    e.add_argument("--bloom", required=True)
    e.add_argument("--shard", help="001..018 (HF から curl で流す)")
    e.add_argument("--src", help="ローカルの PSV を読む (試験用)。 --shard 001 と併用すると先頭規則も効く")
    e.add_argument("--out", required=True)
    e.add_argument("--start-rec", type=int, default=0)
    e.add_argument("--end-rec", type=int, default=0)
    e.add_argument("--keep-frac", type=float, default=1.0, help="未学習のうち鍵で決定的に残す割合")
    e.add_argument("--max-out", type=int, default=0)
    e.add_argument("--drop-mate", action="store_true")
    e.add_argument("--min-free-gb", type=float, default=8.0, help="出力先の空きがこれを切ったら打ち切る")
    e.add_argument("--skip-control", action="store_true")
    e.add_argument("--seen-lo", type=float, default=0.25)
    e.add_argument("--seen-hi", type=float, default=0.60)
    e.set_defaults(func=cmd_extract)
    a = ap.parse_args()
    if a.cmd == "extract" and not (a.shard or a.src):
        sys.exit("--shard か --src のどちらかが要る")
    a.func(a)


if __name__ == "__main__":
    main()
