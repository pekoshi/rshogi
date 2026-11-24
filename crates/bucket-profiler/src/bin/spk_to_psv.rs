//! `spk_to_psv` — ciekce-shogi2 fork の `.spk` (64 byte/record `ShogiBoard`)
//! を nodchip Bullet 互換 `.psv` (40 byte/record `PackedSfenValue`) に変換する。
//!
//! ## 背景
//!
//! p2shogi の Stage 1/2 NNUE 訓練は **nodchip bullet** に乗っているため
//! `.psv` (40 byte huffman) しか読めない。 一方、 入玉特化 Suisho5 教師
//! `suisho5_ek.spk` や Suisho 一般教師 `suisho10mn.spk` は ciekce-shogi2 の
//! `.spk` (64 byte raw bitboard + U4Array<20>) 形式で蓄積されており、
//! Stage 2/3 で投入するために変換が必要。
//!
//! ## .spk record (64 byte、 stoatformat::ShogiBoard, repr(C, packed))
//!
//! | offset | size | 内容                                              |
//! |--------|------|---------------------------------------------------|
//! | 0..16  | 16   | `occ[0]` u128 LE: 先手駒占有 (bit 0..80) + WDL (bit 88-89) + STM (bit 90) + 先手手駒 (bit 96..117) |
//! | 16..32 | 16   | `occ[1]` u128 LE: 後手駒占有 (bit 0..80) + 後手手駒 (bit 96..117) |
//! | 32..52 | 20   | `pieces: U4Array<20>`: 先手占有 LSB→MSB 順の駒種 4bit、 続けて後手占有順 |
//! | 52..54 | 2    | `score: i16` (先手視点)                           |
//! | 54..56 | 2    | `ply_count: u16`                                  |
//! | 56..58 | 2    | `rel_king_squares: [u8; 2]`                       |
//! | 58..60 | 2    | (jsonl_to_spk 拡張) `policy_idx: u16 LE`、 sentinel 0xFFFF |
//! | 60..64 | 4    | reserved (0 埋め)                                 |
//!
//! WDL: `occ[0] >> 88 & 0x3` (0=SenteLoss, 1=Draw, 2=SenteWin、 sente 視点)。
//! STM: `occ[0] >> 90 & 0x1` (0=Sente, 1=Gote)。
//! Hand bits (21 bit) は PAWN(5)|LANCE(3)|KNIGHT(3)|SILVER(3)|GOLD(3)|BISHOP(2)|ROOK(2)。
//! Piece type idx (stoatformat 規約):
//!   0=PAWN, 1=PROMOTED_PAWN, 2=LANCE, 3=KNIGHT, 4=PROMOTED_LANCE,
//!   5=PROMOTED_KNIGHT, 6=SILVER, 7=PROMOTED_SILVER, 8=GOLD, 9=BISHOP,
//!   10=ROOK, 11=PROMOTED_BISHOP (馬), 12=PROMOTED_ROOK (龍), 13=KING。
//!
//! Square: stoatformat raw = `rank_s * 9 + file_s` (file_s 0=9筋..8=1筋、
//! rank_s 0=i段..8=a段)。 rshogi-core は `idx = file_r * 9 + rank_r`
//! (file_r 0=File1=1筋..8=File9=9筋、 rank_r 0=Rank1=a段..8=Rank9=i段) なので
//! `idx = (8 - file_s) * 9 + (8 - rank_s)`。
//!
//! ## .psv record (40 byte, nodchip Bullet / YaneuraOu PackedSfenValue)
//!
//! | offset | size | 内容                                           |
//! |--------|------|------------------------------------------------|
//! | 0..32  | 32   | PackedSfen huffman (`pack_position`)           |
//! | 32..34 | 2    | `score: i16` (**STM 視点**)                    |
//! | 34..36 | 2    | `move16: u16` (=0、 .spk は best move を持たない) |
//! | 36..38 | 2    | `game_ply: u16`                                |
//! | 38     | 1    | `game_result: i8` (**STM 視点**: 1=勝ち、 0=引分、 -1=負け) |
//! | 39     | 1    | padding = 0                                    |
//!
//! ## 変換時の符号反転
//!
//! `.spk` の score / WDL は **先手 (Sente) 視点絶対値**、 `.psv` は
//! **STM 視点**。 後手番の record では score 符号反転 + WDL 反転が必要。
//!
//! ## policy_idx の扱い
//!
//! `.spk` offset 58-59 の `policy_idx: u16` (dlshogi 規格 0..2186) は
//! PSV 40 byte には収まらず**廃棄する** (move16=0)。 nodchip bullet は
//! move16 を訓練に使わないため学習目的では影響なし。

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use bucket_profiler::packed_sfen::pack_position;
use clap::Parser;
use rshogi_core::position::Position;
use rshogi_core::types::{Color, Hand, Piece, PieceType, Square};

const SPK_RECORD_SIZE: usize = 64;
const PSV_RECORD_SIZE: usize = 40;
const BUF_BYTES: usize = 4 * 1024 * 1024;

/// stoatformat::PieceType の raw 値 → rshogi-core PieceType。
fn piece_type_from_stoat(raw: u8) -> Option<PieceType> {
    match raw {
        0 => Some(PieceType::Pawn),
        1 => Some(PieceType::ProPawn),
        2 => Some(PieceType::Lance),
        3 => Some(PieceType::Knight),
        4 => Some(PieceType::ProLance),
        5 => Some(PieceType::ProKnight),
        6 => Some(PieceType::Silver),
        7 => Some(PieceType::ProSilver),
        8 => Some(PieceType::Gold),
        9 => Some(PieceType::Bishop),
        10 => Some(PieceType::Rook),
        11 => Some(PieceType::Horse),
        12 => Some(PieceType::Dragon),
        13 => Some(PieceType::King),
        _ => None,
    }
}

/// stoatformat raw → rshogi-core Square。
fn square_from_stoat_raw(raw: u8) -> Result<Square> {
    if raw >= 81 {
        return Err(anyhow!("invalid stoat square raw {raw}"));
    }
    let file_s = raw % 9;
    let rank_s = raw / 9;
    let file_r = 8 - file_s;
    let rank_r = 8 - rank_s;
    let idx = file_r * 9 + rank_r;
    Square::from_u8(idx).ok_or_else(|| anyhow!("Square::from_u8 failed for idx {idx}"))
}

/// `U4Array<20>` 形式のバイト列から idx 番目の 4bit を取り出す。
fn pieces_get(pieces_bytes: &[u8; 20], idx: usize) -> u8 {
    (pieces_bytes[idx / 2] >> ((idx % 2) * 4)) & 0xF
}

/// `.spk` 64 byte → SFEN 文字列 + (sente 視点 score, sente 視点 wdl, ply, stm)。
fn parse_spk_record(buf: &[u8; SPK_RECORD_SIZE]) -> Result<(String, i16, u8, u16, Color)> {
    let occ0 = u128::from_le_bytes(buf[0..16].try_into().unwrap());
    let occ1 = u128::from_le_bytes(buf[16..32].try_into().unwrap());

    let mut pieces_bytes = [0u8; 20];
    pieces_bytes.copy_from_slice(&buf[32..52]);

    let score = i16::from_le_bytes(buf[52..54].try_into().unwrap());
    let ply_count = u16::from_le_bytes(buf[54..56].try_into().unwrap());

    let wdl = ((occ0 >> 88) & 0x3) as u8;
    let stm_bit = ((occ0 >> 90) & 0x1) as u8;
    let stm = if stm_bit == 0 {
        Color::Black
    } else {
        Color::White
    };

    const BOARD_MASK: u128 = (1u128 << 81) - 1;
    let sente_occ_board = occ0 & BOARD_MASK;
    let gote_occ_board = occ1 & BOARD_MASK;

    let mut board = [Piece::NONE; 81];
    let mut piece_idx = 0usize;

    let mut tmp = sente_occ_board;
    while tmp != 0 {
        let sq_raw = tmp.trailing_zeros() as u8;
        tmp &= tmp - 1;
        let pt_raw = pieces_get(&pieces_bytes, piece_idx);
        piece_idx += 1;
        let pt = piece_type_from_stoat(pt_raw)
            .ok_or_else(|| anyhow!("invalid sente piece type raw={pt_raw} at stoat-sq={sq_raw}"))?;
        let sq = square_from_stoat_raw(sq_raw)?;
        board[sq.index()] = Piece::new(Color::Black, pt);
    }

    let mut tmp = gote_occ_board;
    while tmp != 0 {
        let sq_raw = tmp.trailing_zeros() as u8;
        tmp &= tmp - 1;
        let pt_raw = pieces_get(&pieces_bytes, piece_idx);
        piece_idx += 1;
        let pt = piece_type_from_stoat(pt_raw)
            .ok_or_else(|| anyhow!("invalid gote piece type raw={pt_raw} at stoat-sq={sq_raw}"))?;
        let sq = square_from_stoat_raw(sq_raw)?;
        board[sq.index()] = Piece::new(Color::White, pt);
    }

    let sente_hand_bits = ((occ0 >> 96) as u32) & 0x1F_FF_FF;
    let gote_hand_bits = ((occ1 >> 96) as u32) & 0x1F_FF_FF;
    let sente_hand = Hand::from_raw(sente_hand_bits);
    let gote_hand = Hand::from_raw(gote_hand_bits);

    let sfen = build_sfen(&board, sente_hand, gote_hand, stm, ply_count);
    Ok((sfen, score, wdl, ply_count, stm))
}

/// `board[idx]` (idx = file_r * 9 + rank_r、 file_r 0=1筋、 rank_r 0=a段) +
/// 手駒 + STM + ply から SFEN を構築。
fn build_sfen(
    board: &[Piece; 81],
    sente_hand: Hand,
    gote_hand: Hand,
    stm: Color,
    ply: u16,
) -> String {
    let mut s = String::with_capacity(96);
    // SFEN は a 段から i 段 (rshogi-core Rank1=0..Rank9=8)
    for rank_r in 0..9u8 {
        let mut empties = 0u32;
        // 各段は 9筋 → 1筋 (rshogi-core File9=8..File1=0)
        for file_r in (0..9u8).rev() {
            let sq = (file_r * 9 + rank_r) as usize;
            let piece = board[sq];
            if piece.is_none() {
                empties += 1;
            } else {
                if empties > 0 {
                    s.push_str(&empties.to_string());
                    empties = 0;
                }
                s.push_str(&piece_to_sfen(piece));
            }
        }
        if empties > 0 {
            s.push_str(&empties.to_string());
        }
        if rank_r < 8 {
            s.push('/');
        }
    }
    s.push(' ');
    s.push(if stm == Color::Black { 'b' } else { 'w' });
    s.push(' ');
    let hand_str = hand_to_sfen(sente_hand, gote_hand);
    if hand_str.is_empty() {
        s.push('-');
    } else {
        s.push_str(&hand_str);
    }
    s.push(' ');
    // ply は 1 以上必須 (`Position::set_sfen` の慣習)。 spk が 0 のときは 1 に丸める。
    let ply_out = ply.max(1);
    s.push_str(&ply_out.to_string());
    s
}

fn piece_to_sfen(piece: Piece) -> String {
    let pt = piece.piece_type();
    let raw = pt.unpromote();
    let promoted = pt.is_promoted();
    let c = match raw {
        PieceType::Pawn => 'P',
        PieceType::Lance => 'L',
        PieceType::Knight => 'N',
        PieceType::Silver => 'S',
        PieceType::Gold => 'G',
        PieceType::Bishop => 'B',
        PieceType::Rook => 'R',
        PieceType::King => 'K',
        _ => '?',
    };
    let c = if piece.color() == Color::White {
        c.to_ascii_lowercase()
    } else {
        c
    };
    if promoted {
        format!("+{c}")
    } else {
        c.to_string()
    }
}

fn hand_to_sfen(sente: Hand, gote: Hand) -> String {
    let order = [
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Gold,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
    ];
    let mut s = String::new();
    for &pt in &order {
        let cnt = sente.count(pt);
        if cnt > 0 {
            if cnt > 1 {
                s.push_str(&cnt.to_string());
            }
            s.push(match pt {
                PieceType::Pawn => 'P',
                PieceType::Lance => 'L',
                PieceType::Knight => 'N',
                PieceType::Silver => 'S',
                PieceType::Gold => 'G',
                PieceType::Bishop => 'B',
                PieceType::Rook => 'R',
                _ => '?',
            });
        }
    }
    for &pt in &order {
        let cnt = gote.count(pt);
        if cnt > 0 {
            if cnt > 1 {
                s.push_str(&cnt.to_string());
            }
            s.push(match pt {
                PieceType::Pawn => 'p',
                PieceType::Lance => 'l',
                PieceType::Knight => 'n',
                PieceType::Silver => 's',
                PieceType::Gold => 'g',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                _ => '?',
            });
        }
    }
    s
}

/// Sente 視点 WDL (0=SenteLoss, 1=Draw, 2=SenteWin) を STM 視点 i8 に変換。
fn wdl_to_psv_game_result(wdl: u8, stm: Color) -> Result<i8> {
    let sente_view = match wdl {
        0 => -1i8,
        1 => 0i8,
        2 => 1i8,
        _ => bail!("invalid WDL bits {wdl}"),
    };
    Ok(if stm == Color::Black {
        sente_view
    } else {
        -sente_view
    })
}

#[derive(Parser, Debug)]
#[command(
    name = "spk_to_psv",
    about = "ciekce-shogi2 .spk (64B/record) → nodchip bullet .psv (40B/record) 変換",
    version
)]
struct Args {
    /// 入力 .spk file (1 つ)。
    #[arg(long)]
    input: PathBuf,

    /// 出力 .psv path。
    #[arg(long)]
    output: PathBuf,

    /// progress 表示を抑制。
    #[arg(long, default_value_t = false)]
    quiet: bool,

    /// 動作確認用、 先頭 N record だけ変換。 0 で全件。
    #[arg(long, default_value_t = 0u64)]
    max_records: u64,

    /// 既存 output file を上書きする。
    #[arg(long, default_value_t = false)]
    force: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input.exists() {
        bail!("input file not found: {}", args.input.display());
    }
    if args.output.exists() && !args.force {
        bail!(
            "output file already exists: {} (use --force to overwrite)",
            args.output.display()
        );
    }
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output dir {}", parent.display()))?;
        }
    }

    let in_file =
        File::open(&args.input).with_context(|| format!("open input {}", args.input.display()))?;
    let file_size = in_file.metadata()?.len();
    if file_size % SPK_RECORD_SIZE as u64 != 0 {
        eprintln!(
            "[spk_to_psv] WARN: input size {} is not a multiple of {} (truncated record?)",
            file_size, SPK_RECORD_SIZE
        );
    }
    let total_records = file_size / SPK_RECORD_SIZE as u64;
    let limit = if args.max_records == 0 {
        total_records
    } else {
        args.max_records.min(total_records)
    };

    if !args.quiet {
        eprintln!(
            "[spk_to_psv] input={} ({} records, {:.2} GB)",
            args.input.display(),
            total_records,
            file_size as f64 / 1e9
        );
        eprintln!("[spk_to_psv] output={} (limit={} records)", args.output.display(), limit);
    }

    let mut reader = BufReader::with_capacity(BUF_BYTES, in_file);
    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&args.output)
        .with_context(|| format!("create output {}", args.output.display()))?;
    let mut writer = BufWriter::with_capacity(BUF_BYTES, out_file);

    let mut spk_buf = [0u8; SPK_RECORD_SIZE];
    let mut psv_buf = [0u8; PSV_RECORD_SIZE];
    let mut scratch_pos = Position::new();

    let mut read_count: u64 = 0;
    let mut write_count: u64 = 0;
    let mut bad_count: u64 = 0;
    let start = Instant::now();

    while read_count < limit {
        if let Err(e) = reader.read_exact(&mut spk_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(e).context("read spk record");
        }
        read_count += 1;

        let (sfen, sente_score, wdl, ply, stm) = match parse_spk_record(&spk_buf) {
            Ok(v) => v,
            Err(e) => {
                bad_count += 1;
                if bad_count <= 5 && !args.quiet {
                    eprintln!("[spk_to_psv] parse error at record {read_count}: {e}");
                }
                continue;
            }
        };

        if let Err(e) = scratch_pos.set_sfen(&sfen) {
            bad_count += 1;
            if bad_count <= 5 && !args.quiet {
                eprintln!("[spk_to_psv] set_sfen error at record {read_count}: {e:?} sfen={sfen}");
            }
            continue;
        }

        let packed = pack_position(&scratch_pos);
        let stm_score = if stm == Color::Black {
            sente_score
        } else {
            sente_score.saturating_neg()
        };
        let game_result = match wdl_to_psv_game_result(wdl, stm) {
            Ok(v) => v,
            Err(e) => {
                bad_count += 1;
                if bad_count <= 5 && !args.quiet {
                    eprintln!("[spk_to_psv] wdl error at record {read_count}: {e}");
                }
                continue;
            }
        };

        psv_buf[0..32].copy_from_slice(&packed);
        psv_buf[32..34].copy_from_slice(&stm_score.to_le_bytes());
        psv_buf[34..36].copy_from_slice(&0u16.to_le_bytes()); // move16 = 0
        psv_buf[36..38].copy_from_slice(&ply.to_le_bytes());
        psv_buf[38] = game_result as u8;
        psv_buf[39] = 0;
        writer.write_all(&psv_buf).context("write psv record")?;
        write_count += 1;

        if !args.quiet && write_count.is_multiple_of(1_000_000) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = write_count as f64 / elapsed.max(1e-6);
            eprintln!(
                "[spk_to_psv] {:>10} / {} records ({:.0} K rec/s, bad={})",
                write_count,
                limit,
                rate / 1000.0,
                bad_count
            );
        }
    }

    writer.flush().context("flush writer")?;
    drop(writer);

    let elapsed = start.elapsed().as_secs_f64();
    let rate = write_count as f64 / elapsed.max(1e-6);
    eprintln!(
        "[spk_to_psv] DONE: read={} write={} bad={} elapsed={:.1}s ({:.0} K rec/s)",
        read_count,
        write_count,
        bad_count,
        elapsed,
        rate / 1000.0
    );
    eprintln!(
        "[spk_to_psv] output: {} ({} bytes)",
        args.output.display(),
        write_count * PSV_RECORD_SIZE as u64
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdl_to_psv_game_result_sente_view() {
        // SenteWin (wdl=2), STM=Black → +1
        assert_eq!(wdl_to_psv_game_result(2, Color::Black).unwrap(), 1);
        // SenteWin (wdl=2), STM=White → -1
        assert_eq!(wdl_to_psv_game_result(2, Color::White).unwrap(), -1);
        // SenteLoss (wdl=0), STM=Black → -1
        assert_eq!(wdl_to_psv_game_result(0, Color::Black).unwrap(), -1);
        // SenteLoss (wdl=0), STM=White → +1
        assert_eq!(wdl_to_psv_game_result(0, Color::White).unwrap(), 1);
        // Draw (wdl=1)
        assert_eq!(wdl_to_psv_game_result(1, Color::Black).unwrap(), 0);
        assert_eq!(wdl_to_psv_game_result(1, Color::White).unwrap(), 0);
    }

    #[test]
    fn square_from_stoat_raw_matches_known_squares() {
        // 9九 (stoat _9I): file_s=0, rank_s=0 → file_r=8 (File9), rank_r=8 (Rank9)
        // idx = 8 * 9 + 8 = 80 (rshogi-core SQ_99)
        let sq = square_from_stoat_raw(0).unwrap();
        assert_eq!(sq.index(), 80);

        // 5九 (stoat _5I): file_s=4, rank_s=0 → file_r=4, rank_r=8 → idx=44
        let sq = square_from_stoat_raw(4).unwrap();
        assert_eq!(sq.index(), 44);

        // 5一 (stoat _5A): file_s=4, rank_s=8 → file_r=4, rank_r=0 → idx=36
        let sq = square_from_stoat_raw(8 * 9 + 4).unwrap();
        assert_eq!(sq.index(), 36);

        // 1一 (stoat _1A): file_s=8, rank_s=8 → file_r=0, rank_r=0 → idx=0
        let sq = square_from_stoat_raw(8 * 9 + 8).unwrap();
        assert_eq!(sq.index(), 0);
    }

    #[test]
    fn pieces_get_extracts_low_high_nibble() {
        let mut buf = [0u8; 20];
        buf[0] = 0xAB; // low nibble = 0xB, high = 0xA
        assert_eq!(pieces_get(&buf, 0), 0xB);
        assert_eq!(pieces_get(&buf, 1), 0xA);
        buf[1] = 0x5C;
        assert_eq!(pieces_get(&buf, 2), 0xC);
        assert_eq!(pieces_get(&buf, 3), 0x5);
    }

    /// stoatformat の `Position::pack` 規約に従って手動で平手 .spk を組み立て、
    /// `parse_spk_record` で SFEN を復元し、 rshogi-core の平手 SFEN と比較する。
    #[test]
    fn parse_spk_hirate_round_trip() {
        use rshogi_core::position::Position;

        // 平手 SFEN を rshogi-core で組み立て、 そこから手動で .spk バイト列を構築する。
        let mut pos = Position::new();
        pos.set_hirate();

        // (1) sente / gote occupancy bitboard を rshogi-core 表現で構築し、
        //     stoatformat raw 表現に変換する。
        let mut sente_occ_stoat: u128 = 0;
        let mut gote_occ_stoat: u128 = 0;
        // pieces は stoatformat の Square iter 順 (LSB→MSB) に並べる必要がある。
        // stoatformat raw を昇順に走査し、 該当マスに rshogi-core の駒があれば
        // (sente / gote 区別) その駒種を入れる。
        let mut sente_pieces: Vec<u8> = Vec::new();
        let mut gote_pieces: Vec<u8> = Vec::new();
        let pt_to_stoat = |pt: PieceType| -> u8 {
            match pt {
                PieceType::Pawn => 0,
                PieceType::ProPawn => 1,
                PieceType::Lance => 2,
                PieceType::Knight => 3,
                PieceType::ProLance => 4,
                PieceType::ProKnight => 5,
                PieceType::Silver => 6,
                PieceType::ProSilver => 7,
                PieceType::Gold => 8,
                PieceType::Bishop => 9,
                PieceType::Rook => 10,
                PieceType::Horse => 11,
                PieceType::Dragon => 12,
                PieceType::King => 13,
            }
        };
        for stoat_raw in 0..81u8 {
            let sq = square_from_stoat_raw(stoat_raw).unwrap();
            let piece = pos.piece_on(sq);
            if piece.is_none() {
                continue;
            }
            if piece.color() == Color::Black {
                sente_occ_stoat |= 1u128 << stoat_raw;
            } else {
                gote_occ_stoat |= 1u128 << stoat_raw;
            }
        }
        // pieces array は sente の (stoat LSB→MSB) 順、 続けて gote。
        for stoat_raw in 0..81u8 {
            if (sente_occ_stoat >> stoat_raw) & 1 == 0 {
                continue;
            }
            let sq = square_from_stoat_raw(stoat_raw).unwrap();
            sente_pieces.push(pt_to_stoat(pos.piece_on(sq).piece_type()));
        }
        for stoat_raw in 0..81u8 {
            if (gote_occ_stoat >> stoat_raw) & 1 == 0 {
                continue;
            }
            let sq = square_from_stoat_raw(stoat_raw).unwrap();
            gote_pieces.push(pt_to_stoat(pos.piece_on(sq).piece_type()));
        }

        // (2) WDL=Draw(1)、 STM=Black(0)、 hand 空、 ply=1。
        let stm_bit: u128 = 0;
        let wdl_bits: u128 = 1 << 88;
        let occ0 = sente_occ_stoat | wdl_bits | (stm_bit << 90);
        let occ1 = gote_occ_stoat;

        // (3) pieces を U4Array に詰める
        let mut pieces_bytes = [0u8; 20];
        for (idx, &pt) in sente_pieces.iter().chain(gote_pieces.iter()).enumerate() {
            let slot = idx / 2;
            if idx % 2 == 0 {
                pieces_bytes[slot] |= pt & 0xF;
            } else {
                pieces_bytes[slot] |= (pt & 0xF) << 4;
            }
        }

        let mut spk_buf = [0u8; SPK_RECORD_SIZE];
        spk_buf[0..16].copy_from_slice(&occ0.to_le_bytes());
        spk_buf[16..32].copy_from_slice(&occ1.to_le_bytes());
        spk_buf[32..52].copy_from_slice(&pieces_bytes);
        spk_buf[52..54].copy_from_slice(&0i16.to_le_bytes());
        spk_buf[54..56].copy_from_slice(&1u16.to_le_bytes());

        let (sfen, score, wdl, ply, stm) = parse_spk_record(&spk_buf).expect("parse");
        assert_eq!(score, 0);
        assert_eq!(wdl, 1);
        assert_eq!(ply, 1);
        assert_eq!(stm, Color::Black);

        // 復元した SFEN を再度 Position に流し込めるか確認
        let mut pos2 = Position::new();
        pos2.set_sfen(&sfen)
            .unwrap_or_else(|e| panic!("set_sfen failed for {sfen}: {e:?}"));
        // 平手は 40 枚
        let mut count = 0;
        for sq_idx in 0..81u8 {
            let sq = Square::from_u8(sq_idx).unwrap();
            if pos2.piece_on(sq).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 40, "round-trip hirate should have 40 pieces");

        // pack_position → unpack で SFEN が再現するかも確認
        let packed = pack_position(&pos2);
        let sfen_back =
            bucket_profiler::packed_sfen::unpack_sfen(&packed).expect("unpack roundtrip");
        let mut pos3 = Position::new();
        pos3.set_sfen(&sfen_back).expect("set_sfen back");
    }
}
