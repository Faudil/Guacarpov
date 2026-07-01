import chess
import torch

class LeelaMoveMapper:
    """
    Maps python-chess Move objects to/from the 73-plane AlphaZero/Leela action space representation.
    The action space size is 8x8x73 = 4672 moves, where 8x8 is the source square,
    and 73 represents:
      - 0 to 55: Queen-like moves (8 directions * 7 distances)
      - 56 to 63: Knight moves (8 possible L-shape offsets)
      - 64 to 72: Underpromotions (3 pieces: N, B, R * 3 directions: left, straight, right)
    """
    def __init__(self):
        # 8 directions for Queen moves: N, NE, E, SE, S, SW, W, NW
        self.queen_dirs = [
            (1, 0),   # 0: N
            (1, 1),   # 1: NE
            (0, 1),   # 2: E
            (-1, 1),  # 3: SE
            (-1, 0),  # 4: S
            (-1, -1), # 5: SW
            (0, -1),  # 6: W
            (1, -1)   # 7: NW
        ]
        
        # 8 knight move offsets
        self.knight_offsets = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # Promotion targets for underpromotions: Knight (0), Bishop (1), Rook (2)
        self.promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    def move_to_index(self, move: chess.Move) -> int:
        """
        Converts a chess.Move to a flat index in [0, 4671].
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        r1, f1 = from_sq // 8, from_sq % 8
        r2, f2 = to_sq // 8, to_sq % 8
        
        dr = r2 - r1
        df = f2 - f1
        
        # 1. Check for underpromotions
        if move.promotion is not None and move.promotion in self.promo_pieces:
            promo_idx = self.promo_pieces.index(move.promotion)
            # direction: capture left (df = -1), straight (df = 0), capture right (df = 1)
            # Map df from [-1, 0, 1] to [0, 1, 2]
            dir_idx = df + 1
            if 0 <= dir_idx <= 2:
                plane_idx = 64 + promo_idx * 3 + dir_idx
                return from_sq * 73 + plane_idx
                
        # 2. Check for Knight moves
        if abs(dr) * abs(df) == 2 and abs(dr) + abs(df) == 3:
            if (dr, df) in self.knight_offsets:
                knight_idx = self.knight_offsets.index((dr, df))
                plane_idx = 56 + knight_idx
                return from_sq * 73 + plane_idx
                
        # 3. Check for Queen-like moves
        direction = -1
        distance = 0
        
        if df == 0 and dr > 0:
            direction = 0  # N
            distance = dr
        elif df > 0 and dr > 0 and dr == df:
            direction = 1  # NE
            distance = dr
        elif dr == 0 and df > 0:
            direction = 2  # E
            distance = df
        elif df > 0 and dr < 0 and -dr == df:
            direction = 3  # SE
            distance = df
        elif df == 0 and dr < 0:
            direction = 4  # S
            distance = -dr
        elif df < 0 and dr < 0 and dr == df:
            direction = 5  # SW
            distance = -dr
        elif dr == 0 and df < 0:
            direction = 6  # W
            distance = -df
        elif df < 0 and dr > 0 and dr == -df:
            direction = 7  # NW
            distance = dr
            
        if direction != -1 and 1 <= distance <= 7:
            plane_idx = direction * 7 + (distance - 1)
            return from_sq * 73 + plane_idx
            
        raise ValueError(f"Move {move} from {chess.square_name(from_sq)} to {chess.square_name(to_sq)} is not representable in Leela format")

    def index_to_move(self, index: int, board: chess.Board) -> chess.Move:
        """
        Converts a flat index in [0, 4671] back to a chess.Move.
        """
        from_sq = index // 73
        plane_idx = index % 73
        
        r1, f1 = from_sq // 8, from_sq % 8
        
        # 1. Queen-like moves (plane 0-55)
        if plane_idx < 56:
            direction = plane_idx // 7
            distance = (plane_idx % 7) + 1
            dr, df = self.queen_dirs[direction]
            r2 = r1 + dr * distance
            f2 = f1 + df * distance
            to_sq = r2 * 8 + f2
            
            # Check for Queen promotion
            piece = board.piece_at(from_sq)
            promotion = None
            if piece is not None and piece.piece_type == chess.PAWN:
                if r2 == 7 or r2 == 0:
                    promotion = chess.QUEEN
            return chess.Move(from_sq, to_sq, promotion=promotion)
            
        # 2. Knight moves (plane 56-63)
        elif plane_idx < 64:
            knight_idx = plane_idx - 56
            dr, df = self.knight_offsets[knight_idx]
            r2 = r1 + dr
            f2 = f1 + df
            to_sq = r2 * 8 + f2
            return chess.Move(from_sq, to_sq)
            
        # 3. Underpromotions (plane 64-72)
        else:
            underpromo_idx = plane_idx - 64
            promo_idx = underpromo_idx // 3
            dir_idx = underpromo_idx % 3
            
            promotion = self.promo_pieces[promo_idx]
            df = dir_idx - 1
            
            # Find the active color to determine target rank
            piece = board.piece_at(from_sq)
            if piece is not None and piece.color == chess.BLACK:
                r2 = 0
            else:
                r2 = 7
                
            f2 = f1 + df
            to_sq = r2 * 8 + f2
            return chess.Move(from_sq, to_sq, promotion=promotion)

    def get_legal_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Returns a boolean mask of shape (4672,) where True indicates a legal move.
        """
        mask = torch.zeros(4672, dtype=torch.bool)
        for move in board.legal_moves:
            try:
                idx = self.move_to_index(move)
                mask[idx] = True
            except ValueError:
                # If a move is somehow not representable (should not happen in normal chess)
                continue
        return mask
