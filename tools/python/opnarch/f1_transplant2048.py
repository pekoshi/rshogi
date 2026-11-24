# -*- coding: utf-8 -*-
# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""試験 F2 の準備: 本体 (現職 top3q-102) の FT を、 幅 2048 の OPN (K16 入力 = HalfKA_hmK16 26,064 行 + factoriser 1,629 行
   = 27,693 行、 学習器 shogi_layerstack_K16.exe `--opn-king16`) の半分の列に写して初期重みを作る。 2026-11-15
   設計書 runs/prereg/DESIGN-20261114-OPN2048-FABLE.md 5.1 節。 幅 1024 の版 (F1) は f1_transplant.py。 CPU だけ。

■ 写し方
   - 元: 本体の f32 checkpoint (top3q-102/optimiser_state/weights.bin) の l0w [73,305 x 1024] / l0b [1024]
     (quantised.bin の FT = round(254 * f32) が全要素で成り立つことは f1_transplant.py src で確認済み)。
   - 行 (K16): 学習器 shogi_halfka.rs の king_bucket16 と同じ式 (kb = f*9 + r、 r >= 6 -> f*3 + (r-6)、 他 -> 15)。
     7〜9 段の 15 区分 (K16 区分 0..14) は元の行をそのまま、 区分 15 (1〜6 段の 30 区分) は 30 行の f32 平均
     (opn_static2.py k16_port_ftw と同じ写像を量子化前の f32 で行う)。
   - 列: FT の後は「前半と後半の要素積」(本体は列 i * 列 i+512、 幅 2048 は列 i * 列 i+1024)。
     本体の列 j (j < 512) -> OPN の列 j、 本体の列 j+512 -> OPN の列 j+1024。 対 (j, j+512) は対 (j, j+1024) になる。
   - 写した列 (0..511 / 1024..1535) の factoriser 行 1,629 は 0 (F1 と同じ)。
   - 残りの列 (512..1023 / 1536..2047) は学習器の初期化と同じ分布: l0w は実特徴行も factoriser 行も N(0, sqrt(2/32)) = N(0, 0.25)
     (shogi_layerstack.rs: `new_affine("l0", ..)` + `l0.init_with_effective_input_size(32)` -> acyclib affine.rs
     `InitSettings::Normal { mean: 0, stdev: sqrt(2/32) }` -> rng.rs の正規乱数)、 l0b は 0 (new_affine の bias は Zeroed)。
     乱数は numpy の default_rng(SEED) (学習器の rand::rng() とは別系列だが分布は同じ)。
   - 後段 (l1w / l1b / l1fw / l1fb / l2w / l2b / l3w / l3b) は weights.bin に入れない = 学習器自身の初期化 (F1 と同じ)。

■ weights.bin の並び (幅 2048 K16 の既存 checkpoint opnK2048-8 で `layout` が確かめる)
   [id '\\n'][u64 要素数][f32 x 要素数] の繰り返し。 l0w は行 = 特徴 (factoriser 1,629 行が先、 次に実特徴 26,064 行)、
   1 行 = 2048 列が連続。 学習器の書き出しは coalesced[r] = l0w[1629 + r] + l0w[r % 1629] を round(254 *) で i16。

usage:
   py -3 f1_transplant2048.py layout                     # 見本 opnK2048-8 の weights.bin を学習器の式で量子化 -> その quantised.bin の FT と全要素一致か
   py -3 f1_transplant2048.py build  <out_dir>           # <out_dir>/optimiser_state/weights.bin (l0w, l0b の 2 項目) を書く
   py -3 f1_transplant2048.py verify <out_dir> [N=20000] # 写した列が本体 (K16 に写したもの) と一致するか + 新しい列の分布
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "opnarch"))
from opn_static2 import KB16, k16_index, k16_port_ftw, load_net2, sfen_state, encode_states  # noqa: E402  (IDLE 優先度 / utf-8)
from f1_transplant import index_weights, tensor, q16, main_quant_ft, m2_rows  # noqa: E402

SRC = "E:/shogi/p2shogi/tmp_mob/top3q/top3q-102"              # 現職 (F1 と同じ)
REF = "E:/shogi/p2shogi/tmp_mob/opnK2048/opnK2048-8"           # 並びの見本 (幅 2048 K16 の OPN、 同じ学習器)
IN45, PI, WM = 73305, 1629, 1024                               # 本体
IN16, WO = 16 * PI, 2048                                       # OPN
HM, HO = WM // 2, WO // 2                                      # 本体 512 / OPN 1024 (要素積の相手までの距離)
COLS_T = np.r_[0:HM, HO:HO + HM]                               # 写した列 (OPN 側)
COLS_M = np.r_[0:HM, HM:WM]                                    # 対応する本体の列 (= 0..1023 の順)
COLS_N = np.r_[HM:HO, HO + HM:WO]                              # 新しい列 (乱数)
INIT_STD = float(np.sqrt(np.float32(2.0) / np.float32(32.0)))  # 0.25
SEED = 20261115


def port_f32(l0w_main):
    """本体 f32 l0w [73305*1024] (memmap) -> K16 [26064, 1024] f32。 区分 15 は 30 区分の平均 (f64 で足して f32 に)。"""
    t = np.asarray(l0w_main).reshape(45, PI, WM)
    out = np.zeros((16, PI, WM), dtype=np.float32)
    s = np.zeros((PI, WM), dtype=np.float64); n15 = 0
    for kb in range(45):
        if KB16[kb] < 15:
            out[KB16[kb]] = t[kb]
        else:
            s += t[kb]; n15 += 1
    assert n15 == 30
    out[15] = (s / 30.0).astype(np.float32)
    return out.reshape(IN16, WM)


def coalesce_q(lw):
    """学習器の書き出し (shogi_layerstack.rs ft_weights_leb128 の factorise 分岐) を再現: [26064, 2048] i16。"""
    fac = np.asarray(lw[:PI * WO]).reshape(PI, WO)
    q = np.empty((IN16, WO), dtype=np.int16); CH = 2048
    for r0 in range(0, IN16, CH):
        r1 = min(IN16, r0 + CH)
        real = np.asarray(lw[(PI + r0) * WO:(PI + r1) * WO]).reshape(-1, WO)
        q[r0:r1] = q16(real + fac[np.arange(r0, r1) % PI])          # f32 + f32 -> f64 で 254 倍して丸め
    return q, fac


def cmd_layout():
    t0 = time.time()
    wp = os.path.join(REF, "optimiser_state", "weights.bin"); idx = index_weights(wp)
    print("見本 %s の項目: %s" % (REF, {k: v[0] for k, v in idx.items()}))
    assert idx["l0w"][0] == (PI + IN16) * WO and idx["l0b"][0] == WO, "見本の l0w / l0b の形が 27693 x 2048 でない"
    net = load_net2(os.path.join(REF, "quantised.bin"))
    assert net["k16"] and net["W"] == WO, net["arch"]
    q, _ = coalesce_q(tensor(wp, idx, "l0w"))
    qb = q16(np.asarray(tensor(wp, idx, "l0b")))
    bw = int((q != net["ftw"]).sum()); bb = int((qb != net["ftb"]).sum())
    print("arch: %s" % net["arch"])
    print("見本の weights.bin を「行 = 特徴 / factoriser 1629 行が先 / 1 行 2048 列」と読んで量子化 -> quantised.bin の FT と "
          "重みの食い違い %d / %d、 bias の食い違い %d / %d / %.0fs" % (bw, q.size, bb, WO, time.time() - t0))
    if bw or bb:
        raise SystemExit("🚨weights.bin の並びの読みが合わない")


def cmd_build(out_dir):
    t0 = time.time()
    od = os.path.join(out_dir, "optimiser_state"); dst = os.path.join(od, "weights.bin")
    if os.path.exists(dst):
        raise SystemExit("🚨出力が既にある (上書きしない): %s" % dst)
    ridx = index_weights(os.path.join(REF, "optimiser_state", "weights.bin"))
    assert ridx["l0w"][0] == (PI + IN16) * WO and ridx["l0b"][0] == WO, "見本の l0w / l0b の形が想定と違う: %s" % ridx
    wp = os.path.join(SRC, "optimiser_state", "weights.bin"); idx = index_weights(wp)
    assert idx["l0w"][0] == IN45 * WM and idx["l0b"][0] == WM, "本体の l0w / l0b の形が想定と違う"
    rng = np.random.default_rng(SEED)
    L = np.empty((PI + IN16, WO), dtype=np.float32)
    # 新しい列: 学習器の初期化 N(0, 0.25) を factoriser 行も含む全行に
    L[:, COLS_N] = rng.standard_normal((PI + IN16, len(COLS_N)), dtype=np.float32) * np.float32(INIT_STD)
    # 写した列: factoriser 行 0 / 実特徴行 = 本体の K16 写し
    L[:PI, COLS_T] = 0.0
    L[PI:, COLS_T] = port_f32(tensor(wp, idx, "l0w"))[:, COLS_M]
    B = np.zeros(WO, dtype=np.float32)
    B[COLS_T] = np.asarray(tensor(wp, idx, "l0b"))[COLS_M]
    os.makedirs(od, exist_ok=True)
    with open(dst + ".tmp", "wb") as f:
        f.write(b"l0w\n" + np.uint64(L.size).tobytes() + L.astype("<f4").tobytes())
        f.write(b"l0b\n" + np.uint64(WO).tobytes() + B.astype("<f4").tobytes())
    os.replace(dst + ".tmp", dst)
    with open(os.path.join(out_dir, "SOURCE.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("F2 初期重み (幅 2048 K16): %s の l0w / l0b を K16 に写し (7-9 段 15 区分はそのまま / 他 30 区分は f32 平均)、 "
                "本体の列 j<512 -> OPN j、 j+512 -> OPN j+1024。 写した列の factoriser 行 = 0。 "
                "列 512..1023 / 1536..2047 は N(0, 0.25) (factoriser 行も) / bias 0 (seed %d)。 後段は学習器の初期化。 "
                "作成 runs/opnarch/f1_transplant2048.py build\n" % (wp, SEED))
    print("書いた %s (%d byte、 項目 l0w %d / l0b %d) / %.0fs" % (dst, os.path.getsize(dst), L.size, WO, time.time() - t0))


def cmd_verify(out_dir, n):
    t0 = time.time()
    dst = os.path.join(out_dir, "optimiser_state", "weights.bin")
    didx = index_weights(dst)
    ridx = index_weights(os.path.join(REF, "optimiser_state", "weights.bin"))
    print("書いた weights.bin の項目: %s" % {k: v[0] for k, v in didx.items()})
    for k in didx:
        assert k in ridx and didx[k][0] == ridx[k][0], "項目 %s の要素数が見本 opnK2048-8 と違う" % k
    lw = tensor(dst, didx, "l0w"); lb = np.asarray(tensor(dst, didx, "l0b"))
    q, fac = coalesce_q(lw)
    qb = q16(lb)
    # (1) 写した列: factoriser 行 0 / 量子化後が本体 f32 の K16 写しの量子化と全要素一致
    wp = os.path.join(SRC, "optimiser_state", "weights.bin"); idx = index_weights(wp)
    e16 = q16(port_f32(tensor(wp, idx, "l0w")))                     # [26064, 1024] 本体の列順
    eb = q16(np.asarray(tensor(wp, idx, "l0b")))
    qt = q[:, COLS_T]
    b1 = int((qt != e16).sum()); b1b = int((qb[COLS_T] != eb).sum()); f0 = int((fac[:, COLS_T] != 0).sum())
    print("(1) 写した列 1024 本: factoriser 行の 0 以外 %d / 量子化後と「本体 f32 の K16 写し」の量子化の食い違い 重み %d / %d、 bias %d / %d"
          % (f0, b1, qt.size, b1b, len(COLS_T)))
    # (2) 本体 quantised.bin を k16_port_ftw した表 (J0 の移植テストの写し方) との差: 区分 0..14 は一致、 区分 15 は丸めの順序だけの差
    _, ftb, ftw, _, _ = main_quant_ft()
    p16 = k16_port_ftw(ftw)
    d = np.abs(qt.astype(np.int32) - p16.astype(np.int32))
    d14, d15 = d[:15 * PI], d[15 * PI:]
    print("(2) 本体 quantised.bin の k16_port_ftw との差: 区分 0..14 の食い違い %d (最大 %d) / 区分 15 の食い違い %d / %d (最大 %d) / bias 食い違い %d"
          % (int((d14 != 0).sum()), int(d14.max()), int((d15 != 0).sum()), d15.size, int(d15.max()), int((qb[COLS_T] != ftb).sum())))
    # (3) アキュムレータ (clip 前の i32) を篩の bucket 0-2 の局面で
    R = m2_rows(n)
    enc = encode_states([sfen_state(r["sfen"]) for r in R])
    st = [np.concatenate([[0], np.cumsum(enc["cnt"][v])]) for v in (0, 1)]
    w_p16 = 0; w_45 = 0; w_p16_mean = []; n_own79 = 0; n_view = 0; act_t = []; act_n = []; fail = False
    for v in (0, 1):
        i45 = enc["idx"][v]; i16 = k16_index(i45)
        kb_pos = (i45[st[v][:-1]] // PI)                                     # 視点ごとの玉区分 (全特徴で同じ)
        own79 = KB16[kb_pos] < 15
        n_own79 += int(own79.sum()); n_view += len(R)
        for c0 in range(0, len(R), 500):
            c1 = min(c0 + 500, len(R)); a, z = st[v][c0], st[v][c1]; off = st[v][c0:c1] - a
            ao = np.add.reduceat(q[i16[a:z]], off, axis=0, dtype=np.int32) + qb[None, :].astype(np.int32)
            at = ao[:, COLS_T]
            ap = np.add.reduceat(p16[i16[a:z]], off, axis=0, dtype=np.int32) + ftb[None, :].astype(np.int32)
            a45 = np.add.reduceat(ftw[i45[a:z]], off, axis=0, dtype=np.int32) + ftb[None, :].astype(np.int32)
            dp = np.abs(at - ap); w_p16 = max(w_p16, int(dp.max())); w_p16_mean.append(dp.mean(axis=1))
            m = own79[c0:c1]
            if m.any():
                w_45 = max(w_45, int(np.abs(at[m] - a45[m]).max()))
            # 区分 0..14 の視点では K16 写しの表 (自前) と完全一致が必須
            e = np.add.reduceat(e16[i16[a:z]], off, axis=0, dtype=np.int32) + eb[None, :].astype(np.int32)
            if (at != e).any():
                fail = True
            act_t.append(np.clip(at, 0, 254)); act_n.append(np.clip(ao[:, COLS_N], 0, 254))
    AT = np.concatenate(act_t); AN = np.concatenate(act_n)
    print("(3) アキュムレータ (%d 局面 x 2 視点、 写した列 1024 本):" % len(R))
    print("    自前の K16 写し (量子化) と: %s" % ("全要素一致" if not fail else "🚨不一致あり"))
    print("    本体 quantised.bin の k16_port_ftw と: |差| 最大 %d / 視点ごとの平均 |差| の平均 %.4f" % (w_p16, float(np.concatenate(w_p16_mean).mean())))
    print("    自玉 7〜9 段の視点 (%d / %d = %.1f%%) で本体の元の 45 区分 FT と: |差| 最大 %d (0 なら本体と bit 単位で同じ)"
          % (n_own79, n_view, 100.0 * n_own79 / n_view, w_45))
    print("    clip 後: 写した列 0 の率 %.3f / 254 の率 %.3f / 中央 %.0f   新しい列 0 の率 %.3f / 254 の率 %.3f / 中央 %.0f" % (
        float((AT == 0).mean()), float((AT == 254).mean()), float(np.median(AT)),
        float((AN == 0).mean()), float((AN == 254).mean()), float(np.median(AN))))
    # (4) 新しい列の分布 (学習器の初期化 N(0, 0.25) / bias 0)
    LN = np.asarray(lw).reshape(PI + IN16, WO)[:, COLS_N]
    print("(4) 新しい列 1024 本: l0w 平均 %+.5f / 標準偏差 %.5f (期待 0 / %.5f) / factoriser 行の標準偏差 %.5f / bias の 0 以外 %d"
          % (float(LN.mean()), float(LN.std()), INIT_STD, float(LN[:PI].std()), int((lb[COLS_N] != 0).sum())))
    ok = (not fail and b1 == 0 and b1b == 0 and f0 == 0 and w_45 == 0 and int(d14.max()) == 0
          and abs(float(LN.std()) - INIT_STD) < 1e-3 and int((lb[COLS_N] != 0).sum()) == 0)
    if not ok:
        raise SystemExit("🚨写しが想定と一致しない")
    print("OK / %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else ""
    if m == "layout":
        cmd_layout()
    elif m == "build":
        cmd_build(sys.argv[2])
    elif m == "verify":
        cmd_verify(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
    else:
        raise SystemExit(__doc__)
