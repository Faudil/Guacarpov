"""
JEPA v3 Data Pipeline — Trajectory Based Storage for Spatio-Temporal and Deep architectures.
Stores full game trajectories in a highly compressed format (npz with structural arrays) to 
satisfy <100GB constraints for 10M+ games.

Provides dataset utilities to dynamically reconstruct the 111-channel AlphaZero style tensors.
"""

import chess
import chess.pgn
import io
import os
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
import torch

# Add script directory to system path for robust imports
import sys
sys.path.append(os.path.dirname(__file__))
jepa_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'JEPA_Policy')
sys.path.append(jepa_policy_path)
from data_utils import compact_state_dtype, board_to_compact_state
from leela_move_mapper import LeelaMoveMapper

def parse_result(result_str: str) -> float:
    if result_str == '1-0': return 1.0
    elif result_str == '0-1': return -1.0
    else: return 0.0

def parse_and_replay_game_worker(args_tuple):
    movetext, result_str = args_tuple
    try:
        pgn = io.StringIO(movetext)
        game = chess.pgn.read_game(pgn)
        if game is None:
            return None
        
        mapper = LeelaMoveMapper()
        board = game.board()
        history = []
        
        states = [board_to_compact_state(board, history)]
        for move in game.mainline_moves():
            try:
                action_idx = mapper.move_to_index(move)
            except ValueError:
                action_idx = -1
            states[-1]['action'] = action_idx
            
            board.push(move)
            states.append(board_to_compact_state(board, history))
            
        if len(states) < 2:
            return None
            
        outcome = parse_result(result_str)
        return np.array(states, dtype=compact_state_dtype), outcome
    except Exception:
        return None

def collect_jepa_data(args):
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, 'metadata.pt')
    
    games_processed = 0
    raw_games_seen = 0
    chunk_idx = 0
    
    if os.path.exists(metadata_path):
        meta = torch.load(metadata_path, map_location='cpu', weights_only=True)
        if meta.get('min_elo', 0) == args.min_elo:
            games_processed = meta.get('games_processed', 0)
            raw_games_seen = meta.get('raw_games_seen', 0)
            chunk_idx = meta.get('num_chunks', 0)
            print(f"▶️ Resuming from raw game index {raw_games_seen} (games processed: {games_processed}, chunks: {chunk_idx}).")
            
    print(f"Streaming from 'Lichess/standard-chess-games' (min_elo={args.min_elo})...")
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    if raw_games_seen > 0: ds = ds.skip(raw_games_seen)
        
    chunk_states = []
    chunk_lengths = []
    chunk_outcomes = []
    
    pbar = tqdm(total=args.num_games, initial=games_processed, desc="Collecting games")
    
    def raw_game_generator():
        nonlocal raw_games_seen
        for game_data in ds:
            if games_processed >= args.num_games: break
            raw_games_seen += 1
            try:
                w_elo = int(game_data.get('WhiteElo', 0) or 0)
                b_elo = int(game_data.get('BlackElo', 0) or 0)
            except (ValueError, TypeError): continue
            
            if w_elo < args.min_elo or b_elo < args.min_elo: continue
            
            result_str = game_data.get('Result', '*')
            if result_str not in ('1-0', '0-1', '1/2-1/2'): continue
            
            movetext = game_data.get('movetext', '')
            if not movetext: continue
            yield (movetext, result_str)
            
    num_workers = args.num_workers if args.num_workers > 0 else mp.cpu_count()
    
    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap(parse_and_replay_game_worker, raw_game_generator(), chunksize=100):
            if res is None: continue
            
            states, outcome = res
            chunk_states.append(states)
            chunk_lengths.append(len(states))
            chunk_outcomes.append(outcome)
            
            games_processed += 1
            pbar.update(1)
            
            if len(chunk_outcomes) >= args.games_per_chunk:
                all_states = np.concatenate(chunk_states)
                all_lengths = np.array(chunk_lengths, dtype=np.int32)
                all_outcomes = np.array(chunk_outcomes, dtype=np.float32)
                
                # Save as highly compressed npz
                out_file = os.path.join(args.output_dir, f'chunk_{chunk_idx}.npz')
                np.savez_compressed(out_file, states=all_states, lengths=all_lengths, outcomes=all_outcomes)
                
                chunk_idx += 1
                chunk_states, chunk_lengths, chunk_outcomes = [], [], []
                
                torch.save({
                    'games_processed': games_processed,
                    'raw_games_seen': raw_games_seen,
                    'num_chunks': chunk_idx,
                    'min_elo': args.min_elo
                }, metadata_path)
                
    pbar.close()
    
    if chunk_outcomes:
        all_states = np.concatenate(chunk_states)
        all_lengths = np.array(chunk_lengths, dtype=np.int32)
        all_outcomes = np.array(chunk_outcomes, dtype=np.float32)
        out_file = os.path.join(args.output_dir, f'chunk_{chunk_idx}.npz')
        np.savez_compressed(out_file, states=all_states, lengths=all_lengths, outcomes=all_outcomes)
        chunk_idx += 1
        
    torch.save({
        'games_processed': games_processed,
        'raw_games_seen': raw_games_seen,
        'num_chunks': chunk_idx,
        'min_elo': args.min_elo
    }, metadata_path)
    
    print(f"\n✅ Data collection complete! Saved {chunk_idx} compressed chunks.")

# --- DataLoader Utilities for the 111-channel Tensors ---

def reconstruct_111_channel_state(trajectory: np.ndarray, current_step: int) -> torch.Tensor:
    """
    Given a game trajectory (array of compact_state_dtype) and the current step t,
    reconstructs the 111-channel AlphaZero style tensor.
    
    Channels:
    0-95: 12 piece placements for steps t, t-1, t-2, ..., t-7
    96: Turn (all 1s or 0s)
    97: White Kingside castling
    98: White Queenside castling
    99: Black Kingside castling
    100: Black Queenside castling
    101: En passant square
    102: Halfmove clock
    103-110: Repetition counts (1hot encoding for 0-7 repetitions)
    """
    tensor = torch.zeros((111, 8, 8), dtype=torch.float32)
    
    # 1. Piece History (96 channels)
    for hist_idx in range(8):
        t = current_step - hist_idx
        if t < 0: t = 0 # Padding by copying the initial state if we go past start of game
        
        board_state = trajectory['board'][t] # (64,)
        base_channel = hist_idx * 12
        for c in range(12):
            mask = (board_state == (c + 1)).reshape(8, 8)
            tensor[base_channel + c, mask] = 1.0
            
    # Current step rule states
    curr_state = trajectory[current_step]
    
    # 2. Turn (1 channel)
    tensor[96, :, :] = float(curr_state['turn'])
    
    # 3. Castling (4 channels)
    castling = curr_state['castling']
    tensor[97, :, :] = float((castling & 1) > 0)
    tensor[98, :, :] = float((castling & 2) > 0)
    tensor[99, :, :] = float((castling & 4) > 0)
    tensor[100, :, :] = float((castling & 8) > 0)
    
    # 4. En passant (1 channel)
    ep = curr_state['en_passant']
    if ep != -1:
        r, c = ep // 8, ep % 8
        tensor[101, r, c] = 1.0
        
    # 5. Halfmove clock (1 channel, scaled to [0, 1])
    tensor[102, :, :] = float(curr_state['halfmove']) / 100.0
    
    # 6. Repetition (8 channels, 1-hot)
    rep = min(int(curr_state['repetition']), 7)
    tensor[103 + rep, :, :] = 1.0
    
    return tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare JEPA v3 dataset from Lichess games.")
    parser.add_argument('--num_games', type=int, default=30_000_000)
    parser.add_argument('--min_elo', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='jepa_v3_data')
    parser.add_argument('--games_per_chunk', type=int, default=50_000)
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    collect_jepa_data(args)
