# p2shogi 公開版 (2026-11-24): 元は開発環境のスクリプト。 データ・exe・net の絶対パスは各自の環境に合わせて書き換えてください。
"""Bullet v19 ckpt Python inference with YaneuraOu HCP decoder.

Self-contained inference (no cshogi for board state):
  PSV record (172 byte) → unpack YaneuraOu HCP → board state
  → HalfKA_hm sparse encoder
  → forward through saved weights
  → argmax policy logits → predicted move label
  → compare to target's top-1 move label (from visit dist) → accuracy
"""
from __future__ import annotations
import numpy as np
import struct
import sys

# Architecture constants
INPUT_DIM = 73305
NUM_KING_BUCKETS = 45
PIECE_INPUTS = 1629
FE_HAND_END = 90
E_KING_BASE = 1629
L1 = 1536
L2 = 32
L3 = 96
POLICY_HIDDEN = 1536
POLICY_DIMS = 2187
SCALE = 600.0

# Piece type indices (yaneuraou): 0=none, 1=pawn, 2=lance, 3=knight, 4=silver,
# 5=bishop, 6=rook, 7=gold, 8=king, 9=propawn, 10=prolance, 11=proknight, 12=prosilver,
# 13=horse, 14=dragon
PT_NONE, PT_PAWN, PT_LANCE, PT_KNIGHT, PT_SILVER, PT_BISHOP, PT_ROOK, PT_GOLD, PT_KING = range(9)
PT_PROPAWN, PT_PROLANCE, PT_PROKNIGHT, PT_PROSILVER, PT_HORSE, PT_DRAGON = range(9, 15)

COLOR_BLACK, COLOR_WHITE = 0, 1


# ---------------------------------------------------------------------------
# YaneuraOu HCP Huffman decoder
# ---------------------------------------------------------------------------

HUFFMAN_TABLE = [
    (0x00, 1),  # 0 NO_PIECE
    (0x01, 2),  # 1 PAWN
    (0x03, 4),  # 2 LANCE
    (0x0b, 4),  # 3 KNIGHT
    (0x07, 4),  # 4 SILVER
    (0x1f, 6),  # 5 BISHOP
    (0x3f, 6),  # 6 ROOK
    (0x0f, 5),  # 7 GOLD
]
HUFFMAN_IDX_TO_PT = [PT_NONE, PT_PAWN, PT_LANCE, PT_KNIGHT, PT_SILVER, PT_BISHOP, PT_ROOK, PT_GOLD]
HUFFMAN_GOLD_IDX = 7


class BitStream:
    """LSB-first bit stream (matches Bullet's BitStream)."""
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 0

    def read_bit(self) -> int:
        byte_idx = self.cursor // 8
        bit_idx = self.cursor % 8
        self.cursor += 1
        if byte_idx >= len(self.data):
            return 0
        return (self.data[byte_idx] >> bit_idx) & 1

    def read_bits(self, n: int) -> int:
        v = 0
        for i in range(n):
            v |= (self.read_bit() << i)
        return v


def promote_pt(pt: int) -> int:
    """Promote a piece type."""
    if pt == PT_PAWN:
        return PT_PROPAWN
    if pt == PT_LANCE:
        return PT_PROLANCE
    if pt == PT_KNIGHT:
        return PT_PROKNIGHT
    if pt == PT_SILVER:
        return PT_PROSILVER
    if pt == PT_BISHOP:
        return PT_HORSE
    if pt == PT_ROOK:
        return PT_DRAGON
    return pt


def decode_board_piece(stream: BitStream) -> tuple[int, int]:
    """Returns (color, pt). (0, PT_NONE) for empty.
    Reads variable bits: huffman code + promote bit (non-gold) + color bit.
    """
    code = 0
    bits = 0
    while bits <= 6:
        code |= stream.read_bit() << bits
        bits += 1
        for idx, (pattern, length) in enumerate(HUFFMAN_TABLE):
            if bits == length and code == pattern:
                if idx == 0:
                    return (0, PT_NONE)
                base_pt = HUFFMAN_IDX_TO_PT[idx]
                promoted = stream.read_bit() if idx != HUFFMAN_GOLD_IDX else 0
                color = stream.read_bit()  # 0=Black, 1=White
                pt = promote_pt(base_pt) if promoted else base_pt
                return (color, pt)
    return (0, PT_NONE)


def decode_hand_piece(stream: BitStream) -> tuple[int, int, bool]:
    """Returns (color, pt, is_piecebox)."""
    code = 0
    bits = 0
    while bits <= 5:
        code |= stream.read_bit() << bits
        bits += 1
        for idx, (pattern, length) in enumerate(HUFFMAN_TABLE):
            if idx == 0:
                continue
            hand_pattern = pattern >> 1
            hand_len = length - 1
            if bits == hand_len and code == hand_pattern:
                base_pt = HUFFMAN_IDX_TO_PT[idx]
                # promote bit for non-gold (in hand context: is_piecebox)
                is_piecebox = bool(stream.read_bit()) if idx != HUFFMAN_GOLD_IDX else False
                color = stream.read_bit()
                return (color, base_pt, is_piecebox)
    return (0, PT_NONE, False)


def unpack_yo_hcp(hcp: bytes) -> dict:
    """Unpack YaneuraOu HCP (32 bytes) into board state.

    Returns:
      {
        'board': [(color, pt)] * 81,  # (0, PT_NONE) for empty
        'black_hand': {pt: count},
        'white_hand': {pt: count},
        'side_to_move': 0 or 1,
        'black_king_sq': int,
        'white_king_sq': int,
      }
    """
    stream = BitStream(hcp)
    side_to_move = stream.read_bit()
    black_king_sq = stream.read_bits(7)
    white_king_sq = stream.read_bits(7)

    board = [(0, PT_NONE)] * 81
    if black_king_sq < 81:
        board[black_king_sq] = (COLOR_BLACK, PT_KING)
    if white_king_sq < 81:
        board[white_king_sq] = (COLOR_WHITE, PT_KING)

    for sq in range(81):
        if sq == black_king_sq or sq == white_king_sq:
            continue
        color, pt = decode_board_piece(stream)
        board[sq] = (color, pt)

    black_hand = {pt: 0 for pt in range(1, 8)}
    white_hand = {pt: 0 for pt in range(1, 8)}

    MAX_ITER = 100
    iters = 0
    while stream.cursor < 256 and iters < MAX_ITER:
        iters += 1
        color, pt, is_piecebox = decode_hand_piece(stream)
        if pt == PT_NONE:
            break
        if is_piecebox:
            continue
        if color == COLOR_BLACK:
            black_hand[pt] = black_hand.get(pt, 0) + 1
        else:
            white_hand[pt] = white_hand.get(pt, 0) + 1

    return {
        'board': board,
        'black_hand': black_hand,
        'white_hand': white_hand,
        'side_to_move': side_to_move,
        'black_king_sq': black_king_sq,
        'white_king_sq': white_king_sq,
    }


# ---------------------------------------------------------------------------
# HalfKA_hm sparse encoder (port from shogi_halfka.rs)
# ---------------------------------------------------------------------------

# Hand BonaPiece base offsets
F_HAND_BASE = {PT_PAWN: 1, PT_LANCE: 39, PT_KNIGHT: 49, PT_SILVER: 59,
               PT_GOLD: 69, PT_BISHOP: 79, PT_ROOK: 85}
E_HAND_BASE = {PT_PAWN: 20, PT_LANCE: 44, PT_KNIGHT: 54, PT_SILVER: 64,
               PT_GOLD: 74, PT_BISHOP: 82, PT_ROOK: 88}

# Board piece offsets (pt → (enemy_base, friend_base))
PIECE_BASE = {
    PT_PAWN: (171, 90),
    PT_LANCE: (333, 252),
    PT_KNIGHT: (495, 414),
    PT_SILVER: (657, 576),
    PT_BISHOP: (981, 900),
    PT_ROOK: (1305, 1224),
    PT_GOLD: (819, 738),
    PT_PROPAWN: (819, 738),  # Gold equivalent
    PT_PROLANCE: (819, 738),
    PT_PROKNIGHT: (819, 738),
    PT_PROSILVER: (819, 738),
    PT_HORSE: (1143, 1062),
    PT_DRAGON: (1467, 1386),
}
F_KING = 1548
E_KING = 1629  # board piece offset (different from "E_KING_BASE" name above)


def square_inverse(sq: int) -> int:
    return 80 - sq


def king_bucket(king_sq: int, perspective: int) -> int:
    sq = king_sq if perspective == COLOR_BLACK else square_inverse(king_sq)
    file_ = sq // 9
    rank = sq % 9
    file_m = (8 - file_) if file_ >= 5 else file_
    return file_m * 9 + rank


def is_hm_mirror(king_sq: int, perspective: int) -> bool:
    sq = king_sq if perspective == COLOR_BLACK else square_inverse(king_sq)
    return (sq // 9) >= 5


def bonapiece_board(color: int, pt: int, sq: int, perspective: int) -> int:
    if pt not in PIECE_BASE:
        return 0
    is_friend = 1 if color == perspective else 0
    base = PIECE_BASE[pt][is_friend]
    sq_view = sq if perspective == COLOR_BLACK else square_inverse(sq)
    return base + sq_view


def bonapiece_king(king_sq: int, is_friend: bool, perspective: int) -> int:
    sq_view = king_sq if perspective == COLOR_BLACK else square_inverse(king_sq)
    base = F_KING if is_friend else E_KING
    return base + sq_view


def bonapiece_hand(owner_color: int, pt: int, count: int, perspective: int) -> int:
    is_friend = (owner_color == perspective)
    base = F_HAND_BASE.get(pt, 0) if is_friend else E_HAND_BASE.get(pt, 0)
    if base == 0:
        return 0
    return base + count - 1


def pack_bonapiece(bp: int, hm_mirror: bool) -> int:
    pp = bp
    if hm_mirror and pp >= FE_HAND_END:
        rel = pp - FE_HAND_END
        piece_index = rel // 81
        sq = rel % 81
        file_ = sq // 9
        rank = sq % 9
        mirrored_file = 8 - file_
        mirrored_sq = mirrored_file * 9 + rank
        pp = FE_HAND_END + piece_index * 81 + mirrored_sq
    if pp >= E_KING:
        pp -= 81
    return pp


def halfka_index(king_bucket_val: int, packed_bp: int) -> int:
    return king_bucket_val * PIECE_INPUTS + packed_bp


def encode_halfka(state: dict) -> tuple[list[int], list[int]]:
    """Returns (stm_indices, nstm_indices) of active features."""
    stm = state['side_to_move']
    nstm = 1 - stm
    bk_sq = state['black_king_sq']
    wk_sq = state['white_king_sq']
    if bk_sq >= 81 or wk_sq >= 81:
        return [], []

    stm_king_sq = bk_sq if stm == COLOR_BLACK else wk_sq
    nstm_king_sq = wk_sq if stm == COLOR_BLACK else bk_sq

    stm_kb = king_bucket(stm_king_sq, stm)
    stm_hm = is_hm_mirror(stm_king_sq, stm)
    nstm_kb = king_bucket(nstm_king_sq, nstm)
    nstm_hm = is_hm_mirror(nstm_king_sq, nstm)

    stm_idx = []
    nstm_idx = []

    for sq in range(81):
        color, pt = state['board'][sq]
        if pt == PT_NONE or pt == PT_KING:
            continue
        stm_bp = bonapiece_board(color, pt, sq, stm)
        if stm_bp > 0:
            stm_idx.append(halfka_index(stm_kb, pack_bonapiece(stm_bp, stm_hm)))
        nstm_bp = bonapiece_board(color, pt, sq, nstm)
        if nstm_bp > 0:
            nstm_idx.append(halfka_index(nstm_kb, pack_bonapiece(nstm_bp, nstm_hm)))

    # kings (from each perspective: friend + enemy)
    stm_friend = bonapiece_king(stm_king_sq, True, stm)
    stm_enemy = bonapiece_king(nstm_king_sq, False, stm)
    nstm_friend = bonapiece_king(nstm_king_sq, True, nstm)
    nstm_enemy = bonapiece_king(stm_king_sq, False, nstm)
    stm_idx.append(halfka_index(stm_kb, pack_bonapiece(stm_friend, stm_hm)))
    stm_idx.append(halfka_index(stm_kb, pack_bonapiece(stm_enemy, stm_hm)))
    nstm_idx.append(halfka_index(nstm_kb, pack_bonapiece(nstm_friend, nstm_hm)))
    nstm_idx.append(halfka_index(nstm_kb, pack_bonapiece(nstm_enemy, nstm_hm)))

    # hand pieces
    for owner in (COLOR_BLACK, COLOR_WHITE):
        hand = state['black_hand'] if owner == COLOR_BLACK else state['white_hand']
        for pt in range(1, 8):  # PAWN..GOLD
            count = hand.get(pt, 0)
            for i in range(1, count + 1):
                stm_bp = bonapiece_hand(owner, pt, i, stm)
                if stm_bp > 0:
                    stm_idx.append(halfka_index(stm_kb, pack_bonapiece(stm_bp, stm_hm)))
                nstm_bp = bonapiece_hand(owner, pt, i, nstm)
                if nstm_bp > 0:
                    nstm_idx.append(halfka_index(nstm_kb, pack_bonapiece(nstm_bp, nstm_hm)))

    return stm_idx, nstm_idx


# ---------------------------------------------------------------------------
# Weight loading + forward
# ---------------------------------------------------------------------------

def load_weights(raw_path: str):
    w = np.fromfile(raw_path, dtype=np.float32)
    off = 0
    def take(n):
        nonlocal off
        arr = w[off:off + n]
        off += n
        return arr
    l0w = take(L1 * INPUT_DIM).reshape(L1, INPUT_DIM)
    l0b = take(L1)
    l1w = take(L2 * 2 * L1).reshape(2 * L1, L2).T
    l1b = take(L2)
    l2w = take(L3 * L2).reshape(L2, L3).T
    l2b = take(L3)
    v_outw = take(1 * L3).reshape(L3, 1).T
    v_outb = take(1)
    pol_hw = take(POLICY_HIDDEN * 2 * L1).reshape(2 * L1, POLICY_HIDDEN).T
    pol_hb = take(POLICY_HIDDEN)
    pol_ow = take(POLICY_DIMS * POLICY_HIDDEN).reshape(POLICY_HIDDEN, POLICY_DIMS).T
    pol_ob = take(POLICY_DIMS)
    return dict(
        l0w=l0w, l0b=l0b, l1w=l1w, l1b=l1b, l2w=l2w, l2b=l2b,
        v_outw=v_outw, v_outb=v_outb,
        pol_hw=pol_hw, pol_hb=pol_hb, pol_ow=pol_ow, pol_ob=pol_ob,
        used=off, total=len(w),
    )


def screlu(x):
    return np.clip(x, 0, 1) ** 2


def forward(weights, stm_indices, nstm_indices):
    stm_acc = weights['l0b'].copy()
    for idx in stm_indices:
        if 0 <= idx < INPUT_DIM:
            stm_acc += weights['l0w'][:, idx]
    nstm_acc = weights['l0b'].copy()
    for idx in nstm_indices:
        if 0 <= idx < INPUT_DIM:
            nstm_acc += weights['l0w'][:, idx]

    stm_h = screlu(stm_acc)
    nstm_h = screlu(nstm_acc)
    concat = np.concatenate([stm_h, nstm_h])

    h1 = screlu(weights['l1w'] @ concat + weights['l1b'])
    h2 = screlu(weights['l2w'] @ h1 + weights['l2b'])
    value = (weights['v_outw'] @ h2 + weights['v_outb'])[0]

    p_h = screlu(weights['pol_hw'] @ concat + weights['pol_hb'])
    policy_logits = weights['pol_ow'] @ p_h + weights['pol_ob']
    return value, policy_logits


# ---------------------------------------------------------------------------
# dlshogi make_move_label
# ---------------------------------------------------------------------------

def make_move_label(move16: int, color: int) -> int | None:
    to = move16 & 0x7F
    fp = (move16 >> 7) & 0x7F
    promote = (move16 & 0x4000) != 0
    is_white = (color == 1)
    if fp >= 81:
        pt_index = fp - 81
        if not (1 <= pt_index <= 7):
            return None
        hand_piece = pt_index - 1 if pt_index <= 6 else 6
        if to >= 81:
            return None
        to_sq = (80 - to) if is_white else to
        direction = 20 + hand_piece
        return 81 * direction + to_sq
    else:
        if to >= 81 or fp >= 81:
            return None
        to_sq = (80 - to) if is_white else to
        from_sq = (80 - fp) if is_white else fp
        to_file = to_sq // 9
        to_rank = to_sq % 9
        from_file = from_sq // 9
        from_rank = from_sq % 9
        dir_x = from_file - to_file
        dir_y = to_rank - from_rank
        if dir_y < 0 and dir_x == 0:
            direction = 0
        elif dir_y == -2 and dir_x == -1:
            direction = 8
        elif dir_y == -2 and dir_x == 1:
            direction = 9
        elif dir_y < 0 and dir_x < 0:
            direction = 1
        elif dir_y < 0 and dir_x > 0:
            direction = 2
        elif dir_y == 0 and dir_x < 0:
            direction = 3
        elif dir_y == 0 and dir_x > 0:
            direction = 4
        elif dir_y > 0 and dir_x == 0:
            direction = 5
        elif dir_y > 0 and dir_x < 0:
            direction = 6
        else:
            direction = 7
        if promote:
            direction += 10
        return 81 * direction + to_sq


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

RECORD_SIZE = 172
VISITS_OFFSET = 44


def evaluate(raw_path: str, data_path: str, n_samples: int = 1000) -> None:
    print(f"Loading weights from {raw_path}")
    weights = load_weights(raw_path)
    print(f"  weights loaded: {weights['used']}/{weights['total']} f32")

    print(f"\nEvaluating top-1/5 on first {n_samples:,} records of {data_path}")
    top1_correct = 0
    top5_correct = 0
    skipped = 0
    total = 0

    with open(data_path, 'rb') as f:
        for ri in range(n_samples):
            rec = f.read(RECORD_SIZE)
            if len(rec) < RECORD_SIZE:
                break
            hcp = rec[:32]
            try:
                state = unpack_yo_hcp(hcp)
            except Exception as e:
                skipped += 1
                continue

            # target top-1 by visit
            visit_count = struct.unpack_from('<H', rec, 40)[0]
            target_top1_move = None
            best_visit = 0
            for k in range(visit_count):
                base = VISITS_OFFSET + k * 4
                mv = struct.unpack_from('<H', rec, base)[0]
                vs = struct.unpack_from('<H', rec, base + 2)[0]
                if vs > best_visit:
                    best_visit = vs
                    target_top1_move = mv
            if target_top1_move is None:
                skipped += 1
                continue

            stm_idx, nstm_idx = encode_halfka(state)
            if not stm_idx:
                skipped += 1
                continue

            color = state['side_to_move']
            target_label = make_move_label(target_top1_move, color)
            if target_label is None:
                skipped += 1
                continue

            _, policy_logits = forward(weights, stm_idx, nstm_idx)
            top5_pred = np.argsort(-policy_logits)[:5]
            pred_top1 = int(top5_pred[0])

            if pred_top1 == target_label:
                top1_correct += 1
            if target_label in top5_pred:
                top5_correct += 1
            total += 1

            if total % 50 == 0:
                p1 = 100 * top1_correct / total
                p5 = 100 * top5_correct / total
                print(f"  [{total:,}] top1: {p1:.2f}%  top5: {p5:.2f}%")

    if total == 0:
        print("No valid samples!")
        return

    p1 = 100 * top1_correct / total
    p5 = 100 * top5_correct / total
    print(f"\n=== Final ({total:,} samples, {skipped:,} skipped) ===")
    print(f"  Top-1 accuracy: {p1:.2f}%")
    print(f"  Top-5 accuracy: {p5:.2f}%")


if __name__ == "__main__":
    import os
    # arch override via env vars (for v22 L1=2048, etc.)
    if 'L1' in os.environ:
        L1 = int(os.environ['L1'])
    if 'POLICY_HIDDEN' in os.environ:
        POLICY_HIDDEN = int(os.environ['POLICY_HIDDEN'])
    raw = sys.argv[1] if len(sys.argv) > 1 else "E:/shogi/bullet-shogi-nodchip/checkpoints/v19_strict-50/raw.bin"
    data = sys.argv[2] if len(sys.argv) > 2 else "E:/shogi/p2shogi/data/_psv/gct_strict_visits.bin"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    print(f"L1={L1}, POLICY_HIDDEN={POLICY_HIDDEN}")
    evaluate(raw, data, n)
