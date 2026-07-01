import torch
import chess
import sys
import os
import argparse

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA_Policy"))

from jepa_model import ChessJEPA
from policy_model import ChessJepaPolicy

def print_banner():
    print("\n" + "="*60)
    print("         PLAY CHESS AGAINST THE JEPA MODEL           ")
    print("="*60)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load model checkpoint
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
                print(f"✅ AI loaded successfully from '{path}'.")
                loaded_checkpoint = True
                break
            except Exception as e:
                print(f"⚠️ Could not load checkpoint from '{path}': {e}")
                
    if not loaded_checkpoint:
        print("❌ Error: No compatible model checkpoint could be loaded!")
        return
    
    # 2. Interactive setup
    print_banner()
    
    # Choose Color
    user_color_str = ""
    while user_color_str not in ['w', 'b']:
        user_color_str = input("Choose your color (w for White, b for Black): ").strip().lower()
    user_color = chess.WHITE if user_color_str == 'w' else chess.BLACK
    
    # Start Game
    board = chess.Board()
    history = []
    print("\nGame starting! Enter your moves in UCI format (e.g. e2e4, g1f3, e7e8q).")
    
    # Main game loop
    while not board.is_game_over():
        print("\n" + "-"*40)
        # Draw board (White's perspective if player is White, else flipped)
        print(board.unicode(invert_color=True, borders=True))
        print(f"Active turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        if board.turn == user_color:
            # User turn
            move = None
            while move is None:
                user_input = input("\nYour move: ").strip()
                if user_input == 'exit':
                    print("Exiting game. Thanks for playing!")
                    return
                try:
                    candidate_move = chess.Move.from_uci(user_input)
                    if candidate_move in board.legal_moves:
                        move = candidate_move
                    else:
                        print("❌ Illegal move. Try again.")
                except ValueError:
                    print("❌ Invalid input format. Use UCI format (e.g. e2e4).")
            history.append(board.board_fen())
            board.push(move)
        else:
            # AI turn
            print("\nAI is thinking...")
            
            # Display top candidate moves
            top_moves = policy_model.get_move_probabilities(board, history=history, device=device)[:3]
            print("Top 3 moves considered:")
            for rank, (m, prob) in enumerate(top_moves):
                print(f"  {rank+1}. {m} | Prob: {prob:.4f}")
                    
            ai_move = policy_model.propose_move(board, history=history, device=device)
            if ai_move is None or ai_move not in board.legal_moves:
                print("AI returned invalid/no move. Playing random move.")
                ai_move = list(board.legal_moves)[0]
                
            print(f"👉 AI plays: {ai_move}")
            history.append(board.board_fen())
            board.push(ai_move)
            
    # Game Over
    print("\n" + "="*40)
    print("               GAME OVER                     ")
    print("="*40)
    print(board.unicode(invert_color=True, borders=True))
    outcome = board.outcome()
    if outcome:
        if outcome.winner is None:
            print("Game drawn!")
        else:
            winner_str = "White" if outcome.winner == chess.WHITE else "Black"
            print(f"{winner_str} won the game!")
    else:
        print("Game ended.")
    print(f"Final result code: {board.result()}")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play chess interactively against the JEPA AI.")
    parser.add_argument('--model_checkpoint', type=str, default='JEPA_RL/chess_rl_policy.pth', help="Path to policy checkpoint.")
    args = parser.parse_args()
    main(args)
