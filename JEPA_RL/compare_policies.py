import torch
import torch.nn.functional as F
import chess
import numpy as np
import os
import sys
import argparse

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA_Policy"))

from jepa_model import ChessJEPA
from policy_model import ChessJepaPolicy, board_to_tensor

def select_move_sampled(policy, board, history, device, temp=0.5):
    """
    Selects a move by sampling from the policy distribution (with 1-step lookahead adjustments)
    at the given temperature.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
        
    legal_indices = []
    valid_moves = []
    for m in legal_moves:
        try:
            legal_indices.append(policy.move_mapper.move_to_index(m))
            valid_moves.append(m)
        except ValueError:
            continue
            
    if not legal_indices:
        return None
        
    # Forward pass
    fmt = '111' if policy._get_expected_channels() > 12 else '12'
    board_t = board_to_tensor(board, history=history, format=fmt).unsqueeze(0).to(device)
    policy.eval()
    with torch.no_grad():
        logits = policy(board_t)[0]
        latent = policy.jepa_model.forward_context(board_t)
        current_value = policy.jepa_model.predict_value(latent)[0, 0].item()
        
    mask = torch.full_like(logits, float('-inf'))
    mask[legal_indices] = 0.0
    masked_logits = logits + mask
    
    # 1-step lookahead adjustments
    active_value = current_value if board.turn == chess.WHITE else -current_value
    for move, idx in zip(valid_moves, legal_indices):
        next_board = board.copy()
        next_board.push(move)
        if next_board.is_game_over(claim_draw=True):
            if next_board.is_checkmate():
                masked_logits[idx] = 100.0
            else:
                masked_logits[idx] = masked_logits[idx] - (active_value * 15.0)
                
    # Softmax with temperature
    if temp > 0:
        probs = F.softmax(masked_logits / temp, dim=-1)
        try:
            sampled_idx = torch.multinomial(probs, 1).item()
            return policy.move_mapper.index_to_move(sampled_idx, board)
        except RuntimeError:
            best_idx = torch.argmax(masked_logits).item()
            return policy.move_mapper.index_to_move(best_idx, board)
    else:
        # Argmax (deterministic)
        best_idx = torch.argmax(masked_logits).item()
        return policy.move_mapper.index_to_move(best_idx, board)

def play_match(white_policy, black_policy, device, temp=0.5, max_moves=200):
    """
    Plays a single match between two policy models.
    """
    board = chess.Board()
    history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        active_policy = white_policy if board.turn == chess.WHITE else black_policy
        
        # Propose move (uses sampling-based selection with 1-step lookahead)
        move = select_move_sampled(active_policy, board, history, device, temp=temp)
        
        if move is None or move not in board.legal_moves:
            # Fallback to first legal move
            move = list(board.legal_moves)[0]
            
        board.push(move)
        move_count += 1
        
    if board.is_game_over():
        outcome = board.outcome()
        if outcome and outcome.winner is not None:
            return outcome.winner, move_count
        else:
            return None, move_count  # Draw
    else:
        return None, move_count  # Draw due to move limit

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sampling Temperature: {args.temperature}")
    
    # Paths to checkpoints
    sft_path = args.sft_checkpoint
    rl_path = args.rl_checkpoint
    
    if not os.path.exists(sft_path):
        print(f"❌ Error: SFT checkpoint not found at '{sft_path}'")
        return
    if not os.path.exists(rl_path):
        print(f"❌ Error: RL checkpoint not found at '{rl_path}'")
        return
        
    print(f"Loading SFT policy from '{sft_path}'...")
    sft_checkpoint = torch.load(sft_path, map_location=device, weights_only=False)
    sft_jepa_args = sft_checkpoint['jepa_args'] if 'jepa_args' in sft_checkpoint else sft_checkpoint['args']
    
    from factory import build_jepa_model
    jepa_sft = build_jepa_model(sft_jepa_args).to(device)
    sft_head_type = sft_checkpoint.get('head_type', 'linear')
    sft_policy = ChessJepaPolicy(jepa_sft, freeze_jepa=True, head_type=sft_head_type).to(device)
    sft_policy.load_state_dict(sft_checkpoint['model_state_dict'])
    sft_policy.eval()
    
    print(f"Loading RL policy from '{rl_path}'...")
    rl_checkpoint = torch.load(rl_path, map_location=device, weights_only=False)
    rl_jepa_args = rl_checkpoint['jepa_args'] if 'jepa_args' in rl_checkpoint else rl_checkpoint['args']
    
    jepa_rl = build_jepa_model(rl_jepa_args).to(device)
    rl_head_type = rl_checkpoint.get('head_type', 'linear')
    rl_policy = ChessJepaPolicy(jepa_rl, freeze_jepa=True, head_type=rl_head_type).to(device)
    rl_policy.load_state_dict(rl_checkpoint['model_state_dict'])
    rl_policy.eval()
    
    print(f"\nStarting {args.num_games} head-to-head games between SFT and RL policies...")
    
    rl_wins = 0
    sft_wins = 0
    draws = 0
    total_moves = 0
    
    for g in range(args.num_games):
        # Alternate colors: SFT is White on even games, RL is White on odd games
        sft_is_white = (g % 2 == 0)
        
        if sft_is_white:
            white_policy = sft_policy
            black_policy = rl_policy
            white_name, black_name = "SFT", "RL"
        else:
            white_policy = rl_policy
            black_policy = sft_policy
            white_name, black_name = "RL", "SFT"
            
        print(f"  Game {g+1}/{args.num_games}: {white_name} (W) vs {black_name} (B)...", end="", flush=True)
        
        winner, moves = play_match(white_policy, black_policy, device, temp=args.temperature, max_moves=args.max_moves)
        total_moves += moves
        
        if winner == chess.WHITE:
            if sft_is_white:
                print(f" SFT Wins ({moves} moves)")
                sft_wins += 1
            else:
                print(f" RL Wins ({moves} moves)")
                rl_wins += 1
        elif winner == chess.BLACK:
            if sft_is_white:
                print(f" RL Wins ({moves} moves)")
                rl_wins += 1
            else:
                print(f" SFT Wins ({moves} moves)")
                sft_wins += 1
        else:
            print(f" Draw ({moves} moves)")
            draws += 1
            
    print("\n" + "="*50)
    print("             COMPARISON RESULTS             ")
    print("="*50)
    print(f"RL Policy Wins:  {rl_wins} / {args.num_games} ({rl_wins / args.num_games * 100:.1f}%)")
    print(f"SFT Policy Wins: {sft_wins} / {args.num_games} ({sft_wins / args.num_games * 100:.1f}%)")
    print(f"Draws:           {draws} / {args.num_games} ({draws / args.num_games * 100:.1f}%)")
    print(f"Average Game Length: {total_moves / args.num_games:.1f} moves")
    print("-"*50)
    
    improvement = ((rl_wins - sft_wins) / args.num_games) * 100
    print(f"⭐ Net RL Win Rate Advantage: {improvement:+.1f}%")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare SFT and RL policy models directly.")
    parser.add_argument('--sft_checkpoint', type=str, default='JEPA_Policy/chess_sft_policy.pth', help="Path to SFT model checkpoint.")
    parser.add_argument('--rl_checkpoint', type=str, default='JEPA_RL/chess_rl_policy.pth', help="Path to RL model checkpoint.")
    parser.add_argument('--num_games', type=int, default=50, help="Number of games to play.")
    parser.add_argument('--temperature', type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument('--max_moves', type=int, default=200, help="Max moves per game.")
    
    args = parser.parse_args()
    main(args)
