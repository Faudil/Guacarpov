import torch
import chess
import pandas as pd
import os
import random
import numpy as np
import concurrent.futures
from tqdm import tqdm
import argparse

# --- Helper Functions & Constants ---
CSV_PATH = 'chess-evaluations/chessData.csv'
OUTPUT_DIR = 'jepa_processed_data'
CHUNK_SIZE = 100_000  # Size of each chunk saved to disk

piece_to_channel = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}
PAWN_ADVANTAGE_SCALE = 400.0

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(12, 8, 8, dtype=torch.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            rank, file = chess.square_rank(square), chess.square_file(square)
            tensor[channel, rank, file] = 1
    return tensor

def normalize_score(score: str) -> torch.Tensor:
    if score.startswith("#"):
        return torch.tensor(1.0 if score[1] == '+' else -1.0, dtype=torch.float32)
    try:
        scaled_score = float(score) / PAWN_ADVANTAGE_SCALE
        return torch.tanh(torch.tensor(scaled_score, dtype=torch.float32))
    except Exception:
        return torch.tensor(0.0, dtype=torch.float32)

# --- worker function for parallelization ---
def process_sub_chunk_jepa(data):
    fen_list, eval_list = data
    board_tensors = []
    next_board_tensors = []
    starts = []
    ends = []
    promos = []
    evals = []

    for fen, evaluation in zip(fen_list, eval_list):
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                continue
            
            # Select a random legal move
            move = random.choice(legal_moves)
            
            # Convert context board state to tensor
            board_tensor = board_to_tensor(board)
            
            # Copy board, apply move, and convert next board state to tensor
            next_board = board.copy()
            next_board.push(move)
            next_board_tensor = board_to_tensor(next_board)
            
            # Action components
            start_sq = move.from_square
            end_sq = move.to_square
            promo = (move.promotion - 1) if move.promotion is not None else 0
            
            # Value/evaluation grounding
            norm_eval = normalize_score(evaluation)
            
            board_tensors.append(board_tensor)
            next_board_tensors.append(next_board_tensor)
            starts.append(start_sq)
            ends.append(end_sq)
            promos.append(promo)
            evals.append(norm_eval)
        except Exception:
            continue

    if not board_tensors:
        return None

    return (
        torch.stack(board_tensors),
        torch.stack(next_board_tensors),
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(ends, dtype=torch.long),
        torch.tensor(promos, dtype=torch.long),
        torch.stack(evals).view(-1, 1)
    )

def preprocess_jepa_dataset(limit=500_000):
    print(f"Reading '{CSV_PATH}' up to {limit} samples in chunks...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read raw FENs from CSV in batches
    chunk_iterator = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)
    total_samples = 0
    chunk_idx = 0
    
    num_workers = os.cpu_count()
    print(f"Starting parallel transition pre-processing with {num_workers} workers.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_df in chunk_iterator:
            if total_samples >= limit:
                break
                
            fen_series = chunk_df['FEN'].tolist()
            eval_series = chunk_df['Evaluation'].tolist()
            
            # Split current chunk into sub-chunks for the workers
            fen_sub_chunks = np.array_split(fen_series, num_workers)
            eval_sub_chunks = np.array_split(eval_series, num_workers)
            
            jobs = zip(fen_sub_chunks, eval_sub_chunks)
            results = list(executor.map(process_sub_chunk_jepa, jobs))
            
            # Filter None results
            results = [res for res in results if res is not None]
            if not results:
                continue
                
            # Unpack worker results
            boards_list, next_boards_list, starts_list, ends_list, promos_list, evals_list = zip(*results)
            
            # Concatenate chunks
            X_boards = torch.cat(boards_list)
            Y_next_boards = torch.cat(next_boards_list)
            starts = torch.cat(starts_list)
            ends = torch.cat(ends_list)
            promos = torch.cat(promos_list)
            evals = torch.cat(evals_list)
            
            # Truncate if we overshoot the limit
            current_chunk_len = len(X_boards)
            if total_samples + current_chunk_len > limit:
                rem = limit - total_samples
                X_boards = X_boards[:rem].clone()
                Y_next_boards = Y_next_boards[:rem].clone()
                starts = starts[:rem].clone()
                ends = ends[:rem].clone()
                promos = promos[:rem].clone()
                evals = evals[:rem].clone()
                current_chunk_len = rem
                
            total_samples += current_chunk_len
            
            # Save the chunk
            chunk_filename = os.path.join(OUTPUT_DIR, f'chunk_{chunk_idx}.pt')
            torch.save({
                'boards': X_boards,
                'next_boards': Y_next_boards,
                'starts': starts,
                'ends': ends,
                'promos': promos,
                'evals': evals
            }, chunk_filename)
            
            print(f"Saved chunk {chunk_idx} with {current_chunk_len} samples to '{chunk_filename}' (Total: {total_samples}/{limit})")
            chunk_idx += 1
            
    # Save metadata
    metadata = {'total_samples': total_samples, 'chunk_size': CHUNK_SIZE}
    torch.save(metadata, os.path.join(OUTPUT_DIR, 'metadata.pt'))
    print(f"\n✅ Parallel pre-processing complete!")
    print(f"Total samples processed and saved: {total_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess chess FENs into transition pairs for JEPA.")
    parser.add_argument('--limit', type=int, default=500000, help="Number of board states to process.")
    args = parser.parse_args()
    
    preprocess_jepa_dataset(limit=args.limit)
