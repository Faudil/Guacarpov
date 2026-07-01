import os
import argparse
import io
import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
from data_utils import compact_state_dtype, board_to_compact_state
from leela_move_mapper import LeelaMoveMapper
from policy_model import board_to_tensor

def prepare_sft_dataset(args):
    mapper = LeelaMoveMapper()
    
    if args.format == '12':
        output_file = f"JEPA_Policy/sft_cache_format12_g{args.num_games}_elo{args.min_elo}.pt"
    else:
        output_file = f"JEPA_Policy/sft_cache_format111_g{args.num_games}_elo{args.min_elo}.npz"
        
    print(f"Loading HF Lichess dataset 'Lichess/standard-chess-games' in streaming mode...")
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    
    capacity = args.num_games * 85
    
    if args.format == '12':
        states_tensor = torch.zeros((capacity, 12, 8, 8), dtype=torch.uint8)
        actions_tensor = torch.zeros(capacity, dtype=torch.long)
    else:
        states_arr = np.zeros(capacity, dtype=compact_state_dtype)
        actions_arr = np.zeros(capacity, dtype=np.int32)
        game_starts = np.zeros(capacity, dtype=np.int32)
        
    games_processed = 0
    total_moves_collected = 0
    
    pbar = tqdm(total=args.num_games, desc=f"Collecting {args.format}-channel games")
    
    for game_data in ds:
        if games_processed >= args.num_games:
            break
            
        try:
            white_elo = int(game_data.get('WhiteElo', 0) or 0)
            black_elo = int(game_data.get('BlackElo', 0) or 0)
        except Exception:
            continue
            
        if white_elo < args.min_elo or black_elo < args.min_elo:
            continue
            
        movetext = game_data.get('movetext', '')
        if not movetext:
            continue
            
        pgn = io.StringIO(movetext)
        game = chess.pgn.read_game(pgn)
        if game is None:
            continue
            
        board = game.board()
        moves = list(game.mainline_moves())
        history = []
        
        game_start_idx = total_moves_collected
        
        for move in moves:
            try:
                move_idx = mapper.move_to_index(move)
                
                # Dynamic resizing
                if total_moves_collected >= capacity:
                    new_capacity = int(capacity * 1.5)
                    print(f"\nResizing datasets from capacity {capacity} to {new_capacity}...")
                    
                    if args.format == '12':
                        new_states = torch.zeros((new_capacity, 12, 8, 8), dtype=torch.uint8)
                        new_states[:capacity] = states_tensor
                        states_tensor = new_states
                        
                        new_actions = torch.zeros(new_capacity, dtype=torch.long)
                        new_actions[:capacity] = actions_tensor
                        actions_tensor = new_actions
                    else:
                        new_states = np.zeros(new_capacity, dtype=compact_state_dtype)
                        new_states[:capacity] = states_arr
                        states_arr = new_states
                        
                        new_actions = np.zeros(new_capacity, dtype=np.int32)
                        new_actions[:capacity] = actions_arr
                        actions_arr = new_actions
                        
                        new_starts = np.zeros(new_capacity, dtype=np.int32)
                        new_starts[:capacity] = game_starts
                        game_starts = new_starts
                        
                    capacity = new_capacity
                
                if args.format == '12':
                    board_t = board_to_tensor(board)
                    states_tensor[total_moves_collected] = board_t.to(torch.uint8)
                    actions_tensor[total_moves_collected] = move_idx
                else:
                    compact = board_to_compact_state(board, history)
                    states_arr[total_moves_collected] = compact
                    actions_arr[total_moves_collected] = move_idx
                    game_starts[total_moves_collected] = game_start_idx
                    
                total_moves_collected += 1
            except ValueError:
                pass
                
            board.push(move)
            
        games_processed += 1
        pbar.update(1)
        
    pbar.close()
    
    if total_moves_collected == 0:
        print("No moves collected!")
        return
        
    print(f"Successfully collected {total_moves_collected} positions from {games_processed} games.")
    
    if args.format == '12':
        states = states_tensor[:total_moves_collected]
        actions = actions_tensor[:total_moves_collected]
        torch.save({'states': states, 'actions': actions}, output_file)
        print(f"Saved 12-channel dataset to {output_file}")
    else:
        states = states_arr[:total_moves_collected]
        actions = actions_arr[:total_moves_collected]
        starts = game_starts[:total_moves_collected]
        np.savez_compressed(output_file, states=states, actions=actions, game_starts=starts)
        print(f"Saved 111-channel dataset to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', type=int, default=100_000)
    parser.add_argument('--min_elo', type=int, default=2200)
    parser.add_argument('--format', type=str, choices=['12', '111'], default='111')
    args = parser.parse_args()
    prepare_sft_dataset(args)
