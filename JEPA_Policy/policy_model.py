import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from leela_move_mapper import LeelaMoveMapper
from heads import MLPPolicyHead, TransformerPolicyHead, MoEPolicyHead, HybridLatentThinkingHead, MoELatentThinkingHead
# Import ChessJEPA from the JEPA folder
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
from jepa_model import ChessJEPA

piece_to_channel = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

from data_utils import board_to_compact_state, build_111_batch
import numpy as np

def board_to_tensor(board: chess.Board, history=None, format='12') -> torch.Tensor:
    """
    Converts a chess.Board to a (12, 8, 8) or (111, 8, 8) float32 tensor.
    """
    if format == '111':
        if history is None:
            history = []
        compact = board_to_compact_state(board, history.copy())
        compact_arr = np.array([compact])
        indices = np.array([0])
        starts = np.array([0])
        tensor_111 = build_111_batch(indices, compact_arr, starts)
        return tensor_111.squeeze(0)
        
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            rank, file = chess.square_rank(square), chess.square_file(square)
            tensor[channel, rank, file] = 1.0
    return tensor

class ChessJepaPolicy(nn.Module):
    """
    A policy model that wraps the ChessJEPA v2 model to propose chess moves.
    
    Uses 'policy_head' mode: direct policy projection to Leela action space
    (4672-dim logits) + legal move masking.
    """
    VALID_HEAD_TYPES = ('linear', 'mlp', 'transformer', 'moe', 'latent_thinker', 'moe_latent_thinker')

    def __init__(self, jepa_model: ChessJEPA, freeze_jepa=True, head_type='linear'):
        super(ChessJepaPolicy, self).__init__()
        self.jepa_model = jepa_model
        self.move_mapper = LeelaMoveMapper()
        self.head_type = head_type
        
        # Freeze JEPA parameters if requested
        if freeze_jepa:
            for param in self.jepa_model.parameters():
                param.requires_grad = False
                
        # Policy head: maps context latent vector to 4672-dim action space
        latent_dim = self.jepa_model.latent_dim
        if head_type == 'mlp':
            self.policy_head = MLPPolicyHead(latent_dim=latent_dim)
        elif head_type == 'transformer':
            self.policy_head = TransformerPolicyHead(latent_dim=latent_dim)
        elif head_type == 'moe':
            self.policy_head = MoEPolicyHead(latent_dim=latent_dim)
        elif head_type == 'latent_thinker':
            self.policy_head = HybridLatentThinkingHead(latent_dim=latent_dim, jepa_model=self.jepa_model)
        elif head_type == 'moe_latent_thinker':
            self.policy_head = MoELatentThinkingHead(latent_dim=latent_dim, jepa_model=self.jepa_model)
        else:
            self.policy_head = nn.Linear(latent_dim, 4672)

    def _get_expected_channels(self):
        try:
            if hasattr(self.jepa_model, 'context_encoder'):
                ce = self.jepa_model.context_encoder
                if hasattr(ce, 'spatial'): return ce.spatial.stem[0].in_channels
                if hasattr(ce, 'stem'): return ce.stem[0].in_channels
                if hasattr(ce, 'patch_embed'): return ce.patch_embed.proj.in_channels
                if hasattr(ce, 'conv_in'): return ce.conv_in.in_channels
        except Exception:
            pass
        return 111 # Default for v3

    def forward(self, boards):
        """
        Input shape: (B, 12, 8, 8) or (B, 111, 8, 8)
        Outputs logits over the 4672 Leela actions.
        """
        expected_channels = self._get_expected_channels()
        
        # Auto-pad from 12 channels to expected channels (111) to support cached SFT datasets
        if boards.shape[1] < expected_channels:
            B, _, H, W = boards.shape
            padded_boards = torch.zeros((B, expected_channels, H, W), dtype=boards.dtype, device=boards.device)
            padded_boards[:, :boards.shape[1], :, :] = boards
            boards = padded_boards
            
        latents = self.jepa_model.forward_context(boards)
        logits = self.policy_head(latents)
        return logits

    def _propose_move_lookahead(self, board, valid_moves, legal_indices, current_value, masked_logits):
        active_value = current_value if board.turn == chess.WHITE else -current_value
        for move, idx in zip(valid_moves, legal_indices):
            next_board = board.copy()
            next_board.push(move)
            if next_board.is_game_over(claim_draw=True):
                if next_board.is_checkmate():
                    masked_logits[idx] = 100.0
                else:
                    masked_logits[idx] = masked_logits[idx] - (active_value * 15.0)

    def propose_move(self, board: chess.Board, history=None, method='policy_head', device='cpu') -> chess.Move:
        """
        Proposes the best legal move for the given chess.Board.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        if method == 'jepa_predictor':
            move_values = self.get_move_values(board, history=history, device=device)
            return move_values[0][0] if move_values else None
            
        legal_indices, valid_moves = [], []
        for m in legal_moves:
            try:
                legal_indices.append(self.move_mapper.move_to_index(m))
                valid_moves.append(m)
            except ValueError:
                continue
                
        if not legal_indices:
            return None
            
        fmt = '111' if self._get_expected_channels() > 12 else '12'
        board_t = board_to_tensor(board, history=history, format=fmt).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(board_t)[0]
            latent = self.jepa_model.forward_context(board_t)
            current_value = self.jepa_model.predict_value(latent)[0, 0].item()
            
        mask = torch.full_like(logits, float('-inf'))
        mask[legal_indices] = 0.0
        masked_logits = logits + mask
        
        self._propose_move_lookahead(board, valid_moves, legal_indices, current_value, masked_logits)
        
        best_idx = torch.argmax(F.softmax(masked_logits, dim=-1)).item()
        return self.move_mapper.index_to_move(best_idx, board)

    def get_move_values(self, board: chess.Board, history=None, device='cpu') -> list:
        """
        Returns a sorted list of (Move, Value) tuples for all legal moves,
        computed by evaluating the resulting board state using the JEPA value head.
        For terminal states, assigns the exact outcome value.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
            
        move_values = []
        self.eval()
        with torch.no_grad():
            for move in legal_moves:
                next_board = board.copy()
                next_board.push(move)
                
                if next_board.is_game_over(claim_draw=True):
                    if next_board.is_checkmate():
                        value = 1.0 if board.turn == chess.WHITE else -1.0
                    else:
                        value = 0.0
                else:
                    fmt = '111' if self._get_expected_channels() > 12 else '12'
                    next_history = history.copy() + [board.board_fen()] if history is not None else [board.board_fen()]
                    next_board_t = board_to_tensor(next_board, history=next_history, format=fmt).unsqueeze(0).to(device)
                    next_latent = self.jepa_model.forward_context(next_board_t)
                    value = self.jepa_model.predict_value(next_latent)[0, 0].item()
                move_values.append((move, value))
                
        # Sort based on active player turn: maximize for White, minimize for Black
        multiplier = 1.0 if board.turn == chess.WHITE else -1.0
        move_values.sort(key=lambda x: x[1] * multiplier, reverse=True)
        return move_values

    def get_move_probabilities(self, board: chess.Board, history=None, device='cpu') -> list:
        """
        Returns a sorted list of (Move, Probability) tuples for all legal moves.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
            
        legal_indices = []
        valid_moves = []
        for m in legal_moves:
            try:
                legal_indices.append(self.move_mapper.move_to_index(m))
                valid_moves.append(m)
            except ValueError:
                continue
                
        if not legal_indices:
            return []
            
        fmt = '111' if self._get_expected_channels() > 12 else '12'
        board_t = board_to_tensor(board, history=history, format=fmt).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(board_t)[0]
            
        mask = torch.full_like(logits, float('-inf'))
        mask[legal_indices] = 0.0
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        
        move_probs = [(move, probs[idx].item()) for move, idx in zip(valid_moves, legal_indices)]
        move_probs.sort(key=lambda x: x[1], reverse=True)
        return move_probs
