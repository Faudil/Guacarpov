import torch
import chess
import chess.engine
import numpy as np
import os
import sys
import argparse
import random
import math
from tqdm import tqdm

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA_Policy"))

from jepa_model import ChessJEPA
from policy_model import ChessJepaPolicy, board_to_tensor

class RandomPlayer:
    def propose_move(self, board):
        return random.choice(list(board.legal_moves))

class StockfishPlayer:
    def __init__(self, binary_path, elo, search_time=0.05):
        self.binary_path = binary_path
        self.elo = elo
        self.search_time = search_time
        
    def propose_move(self, board):
        # Open a new engine instance for each move to prevent resource leaks
        try:
            with chess.engine.SimpleEngine.popen_uci(self.binary_path) as engine:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
                result = engine.play(board, chess.engine.Limit(time=self.search_time))
                return result.move
        except Exception as e:
            print(f"Error querying Stockfish: {e}. Falling back to random move.")
            return random.choice(list(board.legal_moves))

def play_match(model_player, opponent_player, model_color, device, max_moves=100):
    """
    Plays a single match between the model and an opponent.
    Returns:
        score: Score for the model (1.0 for win, 0.5 for draw, 0.0 for loss)
        moves_count: Number of moves in the game
    """
    board = chess.Board()
    history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        # Determine who to move
        if board.turn == model_color:
            # Model move
            move = model_player.propose_move(board, history=history, device=device)
        else:
            # Opponent move
            if hasattr(opponent_player, 'propose_move'):
                # Stockfish or Random
                if isinstance(opponent_player, StockfishPlayer):
                    move = opponent_player.propose_move(board)
                else:
                    move = opponent_player.propose_move(board)
            else:
                move = random.choice(list(board.legal_moves))
                
        if move is None or move not in board.legal_moves:
            # Fallback to random if engine fails or returns illegal move
            move = random.choice(list(board.legal_moves))
            
        history.append(board.board_fen())
        board.push(move)
        move_count += 1
        
    # Get outcome
    if board.is_game_over():
        outcome = board.outcome()
        if outcome and outcome.winner is not None:
            if outcome.winner == model_color:
                return 1.0, move_count # Win
            else:
                return 0.0, move_count # Loss
        else:
            return 0.5, move_count # Draw
    else:
        return 0.5, move_count # Draw due to limit

def calculate_elo(opponents_stats):
    """
    Calculates estimated ELO using the FIDE Performance Rating (TPR) formula.
    opponents_stats: list of dicts: {'name': name, 'opp_elo': opp_elo, 'wins': w, 'draws': d, 'losses': l}
    """
    total_games = 0
    total_score = 0.0
    weighted_opp_elo_sum = 0.0
    
    for opp in opponents_stats:
        games = opp['wins'] + opp['draws'] + opp['losses']
        if games == 0:
            continue
            
        score = opp['wins'] + 0.5 * opp['draws']
        total_games += games
        total_score += score
        weighted_opp_elo_sum += opp['opp_elo'] * games
        
    if total_games == 0:
        return 400  # Default baseline if no games played
        
    # Average opponent Elo
    r_c = weighted_opp_elo_sum / total_games
    
    # Global win fraction
    p = total_score / total_games
    
    # Clamp win fraction to prevent math domain errors (infinity/NaN) on 100% or 0% win rates
    p = max(0.01, min(0.99, p))
    
    # FIDE Performance Rating (TPR) via logistic inverse
    elo_diff = -400.0 * math.log10((1.0 / p) - 1.0)
    performance_elo = r_c + elo_diff
    
    return int(round(performance_elo))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model Checkpoint
    checkpoint_candidates = []
    if args.model_checkpoint:
        checkpoint_candidates.append(args.model_checkpoint)
    for path in ['JEPA_RL/chess_rl_policy.pth', 'JEPA_Policy/chess_sft_policy.pth', 'JEPA_Policy/chess_policy.pth']:
        if path not in checkpoint_candidates:
            checkpoint_candidates.append(path)
            
    loaded_checkpoint = False
    for path in checkpoint_candidates:
        if os.path.exists(path):
            print(f"Attempting to load model checkpoint from '{path}'...")
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                jepa_args = checkpoint['jepa_args'] if 'jepa_args' in checkpoint else checkpoint['args']
                
                from factory import build_jepa_model
                test_jepa_model = build_jepa_model(jepa_args).to(device)
                
                ckpt_head_type = checkpoint.get('head_type', 'linear')
                test_policy_model = ChessJepaPolicy(test_jepa_model, freeze_jepa=True, head_type=ckpt_head_type).to(device)
                test_policy_model.load_state_dict(checkpoint['model_state_dict'])
                
                jepa_model = test_jepa_model
                policy_model = test_policy_model
                policy_model.eval()
                print(f"✅ Model loaded successfully from '{path}'.")
                loaded_checkpoint = True
                break
            except Exception as e:
                print(f"⚠️ Could not load checkpoint from '{path}': {e}")
                
    if not loaded_checkpoint:
        print("❌ Error: No compatible model checkpoint could be loaded!")
        return
    
    # 2. Define Opponents
    opponents = [
        {
            "name": "Random Player",
            "elo": 400,
            "player": RandomPlayer()
        }
    ]
    
    # Verify if Stockfish exists
    if os.path.exists(args.stockfish_path):
        print(f"✅ Local Stockfish binary found at '{args.stockfish_path}'. Adding Stockfish gauntlet.")
        # Create a gauntlet of Stockfish opponents (Stockfish minimum UCI_Elo is 1320)
        gauntlet_elos = [1320, 1500, 1800, 2100, 2500]
        for elo in gauntlet_elos:
            opponents.append({
                "name": f"Stockfish (Elo {elo})",
                "elo": elo,
                "player": StockfishPlayer(args.stockfish_path, elo=elo)
            })
    else:
        print(f"⚠️ Warning: Stockfish binary not found at '{args.stockfish_path}'. ELO estimation will rely only on Random Player.")
        
    # 3. Play Games
    print(f"\nStarting round-robin tournament ({args.games_per_opponent} games per opponent, alternating colors)...")
    results = []
    
    for opp in opponents:
        print(f"\nPlaying against {opp['name']} (Elo {opp['elo']})...")
        wins, draws, losses = 0, 0, 0
        
        for g in range(args.games_per_opponent):
            # Alternate colors: games 0, 2, 4 as White, 1, 3, 5 as Black
            model_color = chess.WHITE if (g % 2 == 0) else chess.BLACK
            print(f"  Game {g+1}/{args.games_per_opponent}: Model playing as {'White' if model_color == chess.WHITE else 'Black'}...", end="", flush=True)
            
            score, moves = play_match(
                model_player=policy_model,
                opponent_player=opp['player'],
                model_color=model_color,
                device=device,
                max_moves=args.max_moves
            )
            
            if score == 1.0:
                print(f" WIN ({moves} moves)")
                wins += 1
            elif score == 0.0:
                print(f" LOSS ({moves} moves)")
                losses += 1
            else:
                print(f" DRAW ({moves} moves)")
                draws += 1
                
        results.append({
            'name': opp['name'],
            'opp_elo': opp['elo'],
            'wins': wins,
            'draws': draws,
            'losses': losses
        })
        
    # 4. Calculate ELO
    print("\n" + "="*50)
    print("TOURNAMENT RESULTS & ELO ESTIMATION")
    print("="*50)
    for res in results:
        games = res['wins'] + res['draws'] + res['losses']
        score = res['wins'] + 0.5 * res['draws']
        rate = (score / games) * 100 if games > 0 else 0
        print(f"{res['name']} (Elo {res['opp_elo']}): {res['wins']}W / {res['draws']}D / {res['losses']}L | Score rate: {rate:.1f}%")
        
    estimated_elo = calculate_elo(results)
    print("-"*50)
    print(f"⭐ ESTIMATED MODEL ELO: {estimated_elo} ELO")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Chess JEPA Model ELO.")
    parser.add_argument('--model_checkpoint', type=str, default='JEPA_RL/chess_rl_policy.pth', help="Path to model policy checkpoint.")
    parser.add_argument('--stockfish_path', type=str, default='stockfish_bin/stockfish/stockfish-ubuntu-x86-64', help="Path to local Stockfish binary.")
    parser.add_argument('--games_per_opponent', type=int, default=4, help="Number of games to play against each opponent.")
    parser.add_argument('--max_moves', type=int, default=200, help="Max moves per game.")

    
    args = parser.parse_args()
    main(args)
