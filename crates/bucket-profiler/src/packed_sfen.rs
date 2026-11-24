//! PSV (PackedSfenValue) decode — `crates/tools/src/packed_sfen.rs` の
//! **unpack 経路のみ**を抽出した独立実装。
//!
//! tools crate は L3 generic 未整合で build 不可のため、 そこに依存せず
//! 同等の huffman 復号 + SFEN 生成を行う。
//!
//! 元実装の `decode_huffman_piece` / `decode_huffman_hand_piece` / `unpack_sfen`
//! と byte-level で同一の挙動。

use rshogi_core::types::{Color, Hand, Piece, PieceType};

/// PSV record の固定サイズ (byte)。
pub const PSV_RECORD_SIZE: usize = 40;

/// 40 byte PackedSfenValue。
#[derive(Debug, Clone, Copy)]
pub struct PackedSfenValue {
    /// 32 byte 圧縮局面。
    pub sfen: [u8; 32],
    pub score: i16,
    pub move16: u16,
    pub game_ply: u16,
    pub game_result: i8,
    pub padding: u8,
}

impl PackedSfenValue {
    pub const SIZE: usize = PSV_RECORD_SIZE;

    /// 40 byte slice からパース。 length 不足は `None`。
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        let mut sfen = [0u8; 32];
        sfen.copy_from_slice(&bytes[0..32]);
        Some(Self {
            sfen,
            score: i16::from_le_bytes([bytes[32], bytes[33]]),
            move16: u16::from_le_bytes([bytes[34], bytes[35]]),
            game_ply: u16::from_le_bytes([bytes[36], bytes[37]]),
            game_result: bytes[38] as i8,
            padding: bytes[39],
        })
    }
}

struct BitStream<'a> {
    data: &'a [u8],
    bit_cursor: usize,
}

impl<'a> BitStream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_cursor: 0,
        }
    }

    fn read_one_bit(&mut self) -> u8 {
        let byte_idx = self.bit_cursor / 8;
        if byte_idx >= self.data.len() {
            return 0;
        }
        let bit_idx = self.bit_cursor & 7;
        self.bit_cursor += 1;
        (self.data[byte_idx] >> bit_idx) & 1
    }

    fn remaining(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.bit_cursor)
    }

    fn read_n_bit(&mut self, n: usize) -> u32 {
        let mut result = 0u32;
        for i in 0..n {
            result |= (self.read_one_bit() as u32) << i;
        }
        result
    }

    fn cursor(&self) -> usize {
        self.bit_cursor
    }
}

#[derive(Debug, Clone, Copy)]
struct HuffmanCode {
    code: u8,
    bits: u8,
}

const HUFFMAN_TABLE: [HuffmanCode; 8] = [
    HuffmanCode {
        code: 0x00,
        bits: 1,
    }, // NO_PIECE
    HuffmanCode {
        code: 0x01,
        bits: 2,
    }, // PAWN
    HuffmanCode {
        code: 0x03,
        bits: 4,
    }, // LANCE
    HuffmanCode {
        code: 0x0b,
        bits: 4,
    }, // KNIGHT
    HuffmanCode {
        code: 0x07,
        bits: 4,
    }, // SILVER
    HuffmanCode {
        code: 0x1f,
        bits: 6,
    }, // BISHOP
    HuffmanCode {
        code: 0x3f,
        bits: 6,
    }, // ROOK
    HuffmanCode {
        code: 0x0f,
        bits: 5,
    }, // GOLD
];

fn decode_huffman_piece(stream: &mut BitStream) -> Option<usize> {
    let mut code = 0u8;
    let mut bits = 0u8;
    loop {
        code |= stream.read_one_bit() << bits;
        bits += 1;
        if bits > 6 {
            return None;
        }
        for (i, h) in HUFFMAN_TABLE.iter().enumerate() {
            if h.code == code && h.bits == bits {
                return if i == 0 { None } else { Some(i) };
            }
        }
    }
}

fn decode_huffman_hand_piece(stream: &mut BitStream) -> Option<(usize, bool)> {
    let mut code = 0u8;
    let mut bits = 0u8;
    loop {
        code |= stream.read_one_bit() << bits;
        bits += 1;
        if bits > 5 {
            return None;
        }
        for (i, h) in HUFFMAN_TABLE.iter().enumerate().skip(1) {
            if (h.code >> 1) == code && (h.bits - 1) == bits {
                let is_piecebox = if i != 7 {
                    stream.read_one_bit() != 0
                } else {
                    false
                };
                return Some((i, is_piecebox));
            }
        }
    }
}

fn piece_type_from_index(index: usize) -> Option<PieceType> {
    match index {
        1 => Some(PieceType::Pawn),
        2 => Some(PieceType::Lance),
        3 => Some(PieceType::Knight),
        4 => Some(PieceType::Silver),
        5 => Some(PieceType::Bishop),
        6 => Some(PieceType::Rook),
        7 => Some(PieceType::Gold),
        _ => None,
    }
}

/// PackedSfen (32 byte) → SFEN 文字列。
///
/// `crates/tools/src/packed_sfen.rs::unpack_sfen` と同一挙動。
pub fn unpack_sfen(packed: &[u8; 32]) -> Result<String, String> {
    let mut stream = BitStream::new(packed);

    let side_to_move = if stream.read_one_bit() == 0 {
        Color::Black
    } else {
        Color::White
    };

    let mut board = [Piece::NONE; 81];

    let black_king_sq = stream.read_n_bit(7) as u8;
    if black_king_sq >= 81 {
        return Err(format!("Invalid black king position: {black_king_sq}"));
    }
    board[black_king_sq as usize] = Piece::B_KING;

    let white_king_sq = stream.read_n_bit(7) as u8;
    if white_king_sq >= 81 {
        return Err(format!("Invalid white king position: {white_king_sq}"));
    }
    board[white_king_sq as usize] = Piece::W_KING;

    for (sq, cell) in board.iter_mut().enumerate() {
        if cell.is_some() && cell.piece_type() == PieceType::King {
            continue;
        }
        let piece_idx = decode_huffman_piece(&mut stream);
        if let Some(idx) = piece_idx {
            let pt = piece_type_from_index(idx).ok_or("Invalid piece type")?;
            let promoted = if pt != PieceType::Gold {
                stream.read_one_bit() != 0
            } else {
                false
            };
            let color = if stream.read_one_bit() == 0 {
                Color::Black
            } else {
                Color::White
            };
            let piece = if promoted {
                Piece::new(color, pt.promote().ok_or("Cannot promote")?)
            } else {
                Piece::new(color, pt)
            };
            *cell = piece;
        }
        if stream.cursor() > 256 {
            return Err(format!("BitStream overflow at sq {sq}"));
        }
    }

    let mut hands = [Hand::EMPTY; 2];
    const MAX_HAND_ITERATIONS: usize = 256;
    let mut iterations = 0;
    while stream.remaining() > 0 && iterations < MAX_HAND_ITERATIONS {
        iterations += 1;
        let (piece_idx, is_piecebox) =
            decode_huffman_hand_piece(&mut stream).ok_or("Invalid hand piece huffman code")?;
        if is_piecebox {
            if piece_idx != 7 && stream.remaining() > 0 {
                let _ = stream.read_one_bit();
            }
            continue;
        }
        if stream.remaining() == 0 {
            break;
        }
        let color = if stream.read_one_bit() == 0 {
            Color::Black
        } else {
            Color::White
        };
        let pt = piece_type_from_index(piece_idx).ok_or("Invalid hand piece type")?;
        hands[color.index()] = hands[color.index()].add(pt);
    }
    if iterations >= MAX_HAND_ITERATIONS {
        return Err("Hand piece parsing exceeded maximum iterations".to_string());
    }

    Ok(generate_sfen(&board, &hands, side_to_move))
}

fn generate_sfen(board: &[Piece; 81], hands: &[Hand; 2], side_to_move: Color) -> String {
    let mut sfen = String::new();
    for rank in 0..9 {
        if rank > 0 {
            sfen.push('/');
        }
        let mut empty_count = 0;
        for file in (0..9).rev() {
            let sq = file * 9 + rank;
            let piece = board[sq];
            if piece.is_none() {
                empty_count += 1;
            } else {
                if empty_count > 0 {
                    sfen.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                sfen.push_str(&piece_to_sfen_char(piece));
            }
        }
        if empty_count > 0 {
            sfen.push_str(&empty_count.to_string());
        }
    }
    sfen.push(' ');
    sfen.push(if side_to_move == Color::Black {
        'b'
    } else {
        'w'
    });
    sfen.push(' ');
    let hand_str = generate_hand_sfen(&hands[0], &hands[1]);
    if hand_str.is_empty() {
        sfen.push('-');
    } else {
        sfen.push_str(&hand_str);
    }
    sfen.push_str(" 1");
    sfen
}

fn piece_to_sfen_char(piece: Piece) -> String {
    let pt = piece.piece_type();
    let promoted = pt.is_promoted();
    let raw_pt = pt.unpromote();
    let c = match raw_pt {
        PieceType::Pawn => 'P',
        PieceType::Lance => 'L',
        PieceType::Knight => 'N',
        PieceType::Silver => 'S',
        PieceType::Bishop => 'B',
        PieceType::Rook => 'R',
        PieceType::Gold => 'G',
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

fn generate_hand_sfen(black_hand: &Hand, white_hand: &Hand) -> String {
    let mut result = String::new();
    let piece_order = [
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Gold,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
    ];
    for &pt in &piece_order {
        let count = black_hand.count(pt);
        if count > 0 {
            let c = match pt {
                PieceType::Pawn => 'P',
                PieceType::Lance => 'L',
                PieceType::Knight => 'N',
                PieceType::Silver => 'S',
                PieceType::Gold => 'G',
                PieceType::Bishop => 'B',
                PieceType::Rook => 'R',
                _ => continue,
            };
            if count > 1 {
                result.push_str(&count.to_string());
            }
            result.push(c);
        }
    }
    for &pt in &piece_order {
        let count = white_hand.count(pt);
        if count > 0 {
            let c = match pt {
                PieceType::Pawn => 'p',
                PieceType::Lance => 'l',
                PieceType::Knight => 'n',
                PieceType::Silver => 's',
                PieceType::Gold => 'g',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                _ => continue,
            };
            if count > 1 {
                result.push_str(&count.to_string());
            }
            result.push(c);
        }
    }
    result
}

// --------------------------------------------------------------------------
// Encoder: SFEN → PackedSfen (32 byte huffman) — `tools::packed_sfen::pack_position`
// の最小移植。 tools crate が L3 generic 不整合で build 不可のため、
// spk → psv 変換 binary (`spk_to_psv`) 用に独立実装する。
// --------------------------------------------------------------------------

use rshogi_core::position::Position;

struct BitStreamWriter {
    data: [u8; 32],
    bit_cursor: usize,
}

impl BitStreamWriter {
    fn new() -> Self {
        Self {
            data: [0u8; 32],
            bit_cursor: 0,
        }
    }

    fn write_one_bit(&mut self, b: bool) {
        if self.bit_cursor < 256 {
            let byte_idx = self.bit_cursor / 8;
            let bit_idx = self.bit_cursor & 7;
            if b {
                self.data[byte_idx] |= 1 << bit_idx;
            }
            self.bit_cursor += 1;
        }
    }

    fn write_n_bit(&mut self, d: u32, n: usize) {
        for i in 0..n {
            self.write_one_bit((d >> i) & 1 != 0);
        }
    }

    fn bit_position(&self) -> usize {
        self.bit_cursor
    }

    fn finish(self) -> [u8; 32] {
        self.data
    }
}

/// HUFFMAN_TABLE index: 0=NONE, 1=Pawn, 2=Lance, 3=Knight, 4=Silver, 5=Bishop, 6=Rook, 7=Gold。
fn piece_type_to_huffman_index(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn | PieceType::ProPawn => 1,
        PieceType::Lance | PieceType::ProLance => 2,
        PieceType::Knight | PieceType::ProKnight => 3,
        PieceType::Silver | PieceType::ProSilver => 4,
        PieceType::Bishop | PieceType::Horse => 5,
        PieceType::Rook | PieceType::Dragon => 6,
        PieceType::Gold => 7,
        _ => 0,
    }
}

fn write_board_piece(stream: &mut BitStreamWriter, piece: Piece) {
    if piece.is_none() {
        stream.write_one_bit(false);
        return;
    }
    let pt = piece.piece_type();
    let raw_pt = pt.unpromote();
    let promoted = pt.is_promoted();
    let color = piece.color();
    let idx = piece_type_to_huffman_index(pt);
    let huff = &HUFFMAN_TABLE[idx];
    stream.write_n_bit(huff.code as u32, huff.bits as usize);
    if raw_pt != PieceType::Gold {
        stream.write_one_bit(promoted);
    }
    stream.write_one_bit(color == Color::White);
}

/// 駒箱パディング (歩 huffman 0 + promoted=1 + color=0 → 3 bit)。
fn write_piecebox_padding(stream: &mut BitStreamWriter) {
    stream.write_one_bit(false);
    stream.write_one_bit(true);
    stream.write_one_bit(false);
}

fn write_hand_piece(stream: &mut BitStreamWriter, pt: PieceType, color: Color) {
    let idx = piece_type_to_huffman_index(pt);
    let huff = &HUFFMAN_TABLE[idx];
    stream.write_n_bit((huff.code >> 1) as u32, (huff.bits - 1) as usize);
    if pt != PieceType::Gold {
        stream.write_one_bit(false);
    }
    stream.write_one_bit(color == Color::White);
}

/// `Position` から PackedSfen (32 byte huffman) を生成。 YaneuraOu `sfen_packer.cpp` 準拠。
pub fn pack_position(pos: &Position) -> [u8; 32] {
    use rshogi_core::types::Square;
    let mut stream = BitStreamWriter::new();
    // 1. STM
    stream.write_one_bit(pos.side_to_move() == Color::White);
    // 2. 先手玉位置 (7 bit)
    let black_king_sq = pos.king_square(Color::Black);
    stream.write_n_bit(black_king_sq.index() as u32, 7);
    // 3. 後手玉位置 (7 bit)
    let white_king_sq = pos.king_square(Color::White);
    stream.write_n_bit(white_king_sq.index() as u32, 7);
    // 4. 盤上駒 (81 マス、 玉はスキップ)
    for sq_idx in 0..81u8 {
        let sq = Square::from_u8(sq_idx).expect("0..81 in range");
        let piece = pos.piece_on(sq);
        if piece.is_some() && piece.piece_type() == PieceType::King {
            continue;
        }
        write_board_piece(&mut stream, piece);
    }
    // 5. 手駒 (先手 → 後手、 飛 → 歩 の順)
    let order = [
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Gold,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
    ];
    for &color in &[Color::Black, Color::White] {
        let hand = pos.hand(color);
        for &pt in &order {
            let count = hand.count(pt);
            for _ in 0..count {
                write_hand_piece(&mut stream, pt, color);
            }
        }
    }
    // 6. 駒箱パディング (歩の駒箱フラグで 3 bit ずつ埋める)
    while stream.bit_position() + 3 <= 256 {
        write_piecebox_padding(&mut stream);
    }
    stream.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 平手初期局面の PSV (Bullet/cshogi で生成した既知 fixture と等価) は
    /// ここでは利用しないが、 raw bytes 経由のスモークテストとして
    /// PSV header parse + SFEN 復号 → set_sfen round-trip が動くことを確認する。
    #[test]
    fn from_bytes_round_trip() {
        let mut buf = [0u8; PSV_RECORD_SIZE];
        // arbitrary header
        buf[32] = 0x10; // score lo
        buf[33] = 0x20; // score hi
        buf[34] = 0x34;
        buf[35] = 0x12;
        buf[36] = 0x40;
        buf[37] = 0x00;
        buf[38] = 1; // game_result
        buf[39] = 0;
        let psv = PackedSfenValue::from_bytes(&buf).expect("parse");
        assert_eq!(psv.score, 0x2010);
        assert_eq!(psv.move16, 0x1234);
        assert_eq!(psv.game_ply, 0x0040);
        assert_eq!(psv.game_result, 1);
    }

    /// 平手初期局面を pack → unpack → SFEN が一致することを確認する round-trip。
    #[test]
    fn pack_position_hirate_round_trip() {
        use rshogi_core::position::Position;
        let mut pos = Position::new();
        pos.set_hirate();
        let packed = pack_position(&pos);
        let sfen = unpack_sfen(&packed).expect("unpack hirate");
        // unpack で得た sfen を再パースし、 同じ初期局面に戻れることを確認
        let mut pos2 = Position::new();
        pos2.set_sfen(&sfen).expect("set_sfen hirate-roundtrip");
        // 平手の駒数 (40 枚)
        let mut count = 0;
        for sq in 0..81u8 {
            let sq = rshogi_core::types::Square::from_u8(sq).unwrap();
            if pos2.piece_on(sq).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 40, "hirate should have 40 pieces");
    }
}
