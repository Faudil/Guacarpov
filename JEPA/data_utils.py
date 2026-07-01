import chess
import numpy as np
import torch
import torch.nn.functional as F

# Mapping pieces to integer values (0 is empty)
piece_to_int = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2, (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4, (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8, (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10, (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
}

# Define the compact state dtype (~69 bytes per board state)
compact_state_dtype = np.dtype([
    ('board', np.uint8, 64),
    ('castling', np.uint8),      # bitmask: W_K(1), W_Q(2), B_K(4), B_Q(8)
    ('en_passant', np.int8),     # -1 if none, else 0-63
    ('halfmove', np.uint8),
    ('repetition', np.uint8),
    ('turn', np.uint8),          # 1 if white, 0 if black
    ('action', np.int16),        # Leela index of the move taken FROM this state (0-4671). -1 for terminal.
])

def board_to_compact_state(board: chess.Board, history: list) -> np.ndarray:
    """Extracts state from a python-chess board into the compact dtype."""
    arr = np.zeros(1, dtype=compact_state_dtype)
    arr['action'][0] = -1
    
    # 1. Board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            arr['board'][0, square] = piece_to_int[(piece.piece_type, piece.color)]
            
    # 2. Castling
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE): castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE): castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK): castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK): castling |= 8
    arr['castling'][0] = castling
    
    # 3. En passant
    ep = board.ep_square
    arr['en_passant'][0] = ep if ep is not None else -1
    
    # 4. Halfmove clock (Fifty-move rule)
    arr['halfmove'][0] = board.halfmove_clock
    
    # 5. Repetition
    fen_base = board.board_fen()
    reps = sum(1 for h in history if h == fen_base)
    arr['repetition'][0] = reps
    history.append(fen_base)
    
    # 6. Turn
    arr['turn'][0] = 1 if board.turn == chess.WHITE else 0
    
    return arr[0]

def build_111_batch(indices_arr, all_states, game_starts_array):
    """
    Expands an array of compact states into the full 111-channel float tensor.
    indices_arr: Array of indices to expand
    all_states: The full array of compact_state_dtype
    game_starts_array: Array of the same length as all_states indicating the start index of the game for each state
    """
    B = len(indices_arr)
    hist_indices = []
    for h in range(8):
        idx_h = np.maximum(indices_arr - h, game_starts_array[indices_arr])
        hist_indices.append(idx_h)
    hist_indices = np.stack(hist_indices, axis=1)
    
    flat_hist_idx = hist_indices.flatten()
    boards_flat = torch.from_numpy(all_states['board'][flat_hist_idx]).long()
    pieces_one_hot = F.one_hot(boards_flat, num_classes=13)[..., 1:].permute(0, 2, 1).reshape(B, 8, 12, 8, 8).float()
    pieces_hist = pieces_one_hot.reshape(B, 96, 8, 8)
    
    curr_states = all_states[indices_arr]
    
    turns = torch.from_numpy(curr_states['turn']).float().reshape(B, 1, 1, 1).expand(B, 1, 8, 8)
    castling = curr_states['castling']
    c_w_k = torch.from_numpy((castling & 1) > 0).float().reshape(B, 1, 1, 1).expand(B, 1, 8, 8)
    c_w_q = torch.from_numpy((castling & 2) > 0).float().reshape(B, 1, 1, 1).expand(B, 1, 8, 8)
    c_b_k = torch.from_numpy((castling & 4) > 0).float().reshape(B, 1, 1, 1).expand(B, 1, 8, 8)
    c_b_q = torch.from_numpy((castling & 8) > 0).float().reshape(B, 1, 1, 1).expand(B, 1, 8, 8)
    
    ep = torch.from_numpy(curr_states['en_passant']).long()
    ep_plane = torch.zeros(B, 1, 64, dtype=torch.float32)
    valid_ep = ep != -1
    ep_plane[valid_ep, 0, ep[valid_ep]] = 1.0
    ep_plane = ep_plane.reshape(B, 1, 8, 8)
    
    hm = (torch.from_numpy(curr_states['halfmove']).float() / 100.0).reshape(B, 1, 1, 1).expand(B, 1, 8, 8)
    
    rep = torch.from_numpy(curr_states['repetition']).long().clamp(max=7)
    rep_one_hot = F.one_hot(rep, num_classes=8).unsqueeze(-1).unsqueeze(-1).expand(B, 8, 8, 8).float()
    
    curr_feat = torch.cat([turns, c_w_k, c_w_q, c_b_k, c_b_q, ep_plane, hm, rep_one_hot], dim=1)
    
    return torch.cat([pieces_hist, curr_feat], dim=1)
