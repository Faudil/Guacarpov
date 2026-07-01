"""
JEPA v2 Data Pipeline — Streams real chess game trajectories from HuggingFace in parallel.
Optimized for high-throughput using compact numpy IPC transmission and vectorized tensor reconstruction.
"""

import torch
import chess
import chess.pgn
import io
import os
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset

# Mapping pieces to integer values (0 is empty)
piece_to_int = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2, (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4, (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8, (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10, (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
}

def board_to_compact(board: chess.Board) -> np.ndarray:
    """Represent board as a 64-element uint8 array for minimal serialization overhead."""
    arr = np.zeros(64, dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            arr[square] = piece_to_int[(piece.piece_type, piece.color)]
    return arr

def compact_to_tensor_batch(compact_batch: np.ndarray) -> torch.Tensor:
    """Vectorized conversion of a batch of compact board states to 12-channel binary tensors."""
    N = compact_batch.shape[0]
    compact_torch = torch.from_numpy(compact_batch)  # (N, 64)
    grid = compact_torch.view(N, 8, 8)
    
    tensor = torch.zeros((N, 12, 8, 8), dtype=torch.int8)
    for c in range(12):
        tensor[:, c] = (grid == (c + 1)).to(torch.int8)
    return tensor

def parse_result(result_str: str) -> float:
    """Convert PGN result string to numeric outcome."""
    if result_str == '1-0':
        return 1.0   # White wins
    elif result_str == '0-1':
        return -1.0  # Black wins
    else:
        return 0.0   # Draw or unknown

def parse_and_replay_game_worker(args_tuple):
    """Worker target: Replays game on CPU and returns compact numpy representations."""
    movetext, result_str = args_tuple
    try:
        pgn = io.StringIO(movetext)
        game = chess.pgn.read_game(pgn)
        if game is None:
            return None
        
        board = game.board()
        compact_boards = [board_to_compact(board)]
        for move in game.mainline_moves():
            board.push(move)
            compact_boards.append(board_to_compact(board))
            
        if len(compact_boards) < 2:
            return None
            
        outcome = parse_result(result_str)
        return np.array(compact_boards, dtype=np.uint8), outcome
    except Exception:
        return None

def collect_jepa_data(args):
    """
    Streams games from HuggingFace, replays them in parallel using imap, and saves
    transition chunks using vectorized conversions.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, 'metadata.pt')
    
    games_processed = 0
    total_transitions = 0
    raw_games_seen = 0
    chunk_idx = 0
    
    # 1. Load metadata if resuming
    if os.path.exists(metadata_path):
        print(f"Checking existing metadata at '{metadata_path}' for resumption...")
        try:
            meta = torch.load(metadata_path, map_location='cpu', weights_only=True)
            if meta.get('min_elo', 0) == args.min_elo:
                games_processed = meta.get('games_processed', 0)
                total_transitions = meta.get('total_transitions', 0)
                raw_games_seen = meta.get('raw_games_seen', 0)
                chunk_idx = meta.get('num_chunks', 0)
                print(f"▶️ Resuming from raw game index {raw_games_seen} (games processed: {games_processed}, chunks: {chunk_idx}).")
            else:
                print("⚠️ min_elo mismatch in existing metadata. Starting fresh.")
        except Exception as e:
            print(f"⚠️ Failed to load metadata: {e}. Starting fresh.")

    # 2. Load dataset
    print(f"Streaming from 'Lichess/standard-chess-games' (min_elo={args.min_elo})...")
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    
    if raw_games_seen > 0:
        print(f"Skipping first {raw_games_seen} raw games in the HuggingFace stream...")
        ds = ds.skip(raw_games_seen)
        
    current_boards = []
    next_boards = []
    outcomes = []
    
    pbar = tqdm(total=args.num_games, initial=games_processed, desc="Collecting games")
    
    # Define generator for pool feeding (runs on main thread)
    def raw_game_generator():
        nonlocal raw_games_seen
        for game_data in ds:
            if games_processed >= args.num_games:
                break
            raw_games_seen += 1
            
            try:
                white_elo = int(game_data.get('WhiteElo', 0) or 0)
                black_elo = int(game_data.get('BlackElo', 0) or 0)
            except (ValueError, TypeError):
                continue
            
            if white_elo < args.min_elo or black_elo < args.min_elo:
                continue
            
            result_str = game_data.get('Result', '*')
            if result_str not in ('1-0', '0-1', '1/2-1/2'):
                continue
            
            movetext = game_data.get('movetext', '')
            if not movetext:
                continue
                
            yield (movetext, result_str)
            
    num_workers = args.num_workers if args.num_workers > 0 else mp.cpu_count()
    print(f"Initialized Process Pool with {num_workers} parallel workers (Pipelined).")
    
    # 3. Pipelined Processing Loop
    with mp.Pool(processes=num_workers) as pool:
        # Use imap to lazily stream and load-balance games to workers
        for res in pool.imap(parse_and_replay_game_worker, raw_game_generator(), chunksize=64):
            if res is None:
                continue
            
            compact_boards, outcome = res
            T = compact_boards.shape[0]
            
            for t in range(T - 1):
                current_boards.append(compact_boards[t])
                next_boards.append(compact_boards[t + 1])
                outcomes.append(outcome)
                total_transitions += 1
            
            games_processed += 1
            pbar.update(1)
            
            # Save chunk when accumulator is full
            if len(current_boards) >= args.chunk_size:
                current_np = np.stack(current_boards[:args.chunk_size])
                next_np = np.stack(next_boards[:args.chunk_size])
                
                # Perform vectorized conversion to torch on the main thread (extremely fast)
                current_torch = compact_to_tensor_batch(current_np)
                next_torch = compact_to_tensor_batch(next_np)
                
                _save_chunk_tensors(args.output_dir, chunk_idx, current_torch, next_torch, outcomes[:args.chunk_size])
                print(f"  Saved chunk {chunk_idx} ({args.chunk_size} transitions, total: {total_transitions})")
                
                chunk_idx += 1
                current_boards = current_boards[args.chunk_size:]
                next_boards = next_boards[args.chunk_size:]
                outcomes = outcomes[args.chunk_size:]
                
                # Save metadata
                torch.save({
                    'total_transitions': total_transitions,
                    'games_processed': games_processed,
                    'raw_games_seen': raw_games_seen,
                    'num_chunks': chunk_idx,
                    'min_elo': args.min_elo
                }, metadata_path)
            
            # Frequently save metadata progress (e.g. every 500 games) for recovery
            if games_processed % 500 == 0:
                torch.save({
                    'total_transitions': total_transitions,
                    'games_processed': games_processed,
                    'raw_games_seen': raw_games_seen,
                    'num_chunks': chunk_idx,
                    'min_elo': args.min_elo
                }, metadata_path)

    pbar.close()
    
    # Save remaining transitions
    if current_boards:
        current_np = np.stack(current_boards)
        next_np = np.stack(next_boards)
        current_torch = compact_to_tensor_batch(current_np)
        next_torch = compact_to_tensor_batch(next_np)
        
        _save_chunk_tensors(args.output_dir, chunk_idx, current_torch, next_torch, outcomes)
        print(f"  Saved final chunk {chunk_idx} ({len(current_boards)} transitions, total: {total_transitions})")
        chunk_idx += 1
    
    # Save final metadata
    torch.save({
        'total_transitions': total_transitions,
        'games_processed': games_processed,
        'raw_games_seen': raw_games_seen,
        'num_chunks': chunk_idx,
        'min_elo': args.min_elo
    }, metadata_path)
    
    print(f"\n✅ Data collection complete!")
    print(f"   Games processed: {games_processed}")
    print(f"   Raw games scanned: {raw_games_seen}")
    print(f"   Total transitions: {total_transitions}")
    print(f"   Chunks saved: {chunk_idx}")

def _save_chunk_tensors(output_dir, chunk_idx, current_torch, next_torch, outcomes):
    """Save a chunk of transitions to disk."""
    torch.save({
        'boards': current_torch,
        'next_boards': next_torch,
        'outcomes': torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    }, os.path.join(output_dir, f'chunk_{chunk_idx}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare JEPA v2 dataset from Lichess games.")
    parser.add_argument('--num_games', type=int, default=3_000_000, help="Number of games to collect.")
    parser.add_argument('--min_elo', type=int, default=0, help="Minimum Elo for both players.")
    parser.add_argument('--output_dir', type=str, default='jepa_v2_data', help="Output directory for chunks.")
    parser.add_argument('--chunk_size', type=int, default=100_000, help="Transitions per chunk file.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel worker processes (0 = CPU count).")
    
    args = parser.parse_args()
    collect_jepa_data(args)
