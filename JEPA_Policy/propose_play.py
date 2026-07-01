import torch
import chess
import sys
import os

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.dirname(__file__))

from jepa_model import ChessJEPA
from policy_model import ChessJepaPolicy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    jepa_checkpoint_path = "JEPA/chess_jepa.pth"
    policy_checkpoint_path = "JEPA_Policy/chess_policy.pth"
    
    # 1. Load Pre-trained JEPA model
    if not os.path.exists(jepa_checkpoint_path):
        print(f"❌ Error: JEPA checkpoint not found at '{jepa_checkpoint_path}'. Please run JEPA training first!")
        return
        
    from factory import load_jepa_from_checkpoint
    jepa_model, jepa_args = load_jepa_from_checkpoint(jepa_checkpoint_path, device)
    print("✅ JEPA model loaded.")
    
    # 2. Load Policy Model (auto-detect head_type from checkpoint)
    loaded_policy = False
    policy_checkpoint_candidates = [
        "JEPA_Policy/chess_policy.pth",
        "JEPA_Policy/chess_sft_policy.pth",
        "JEPA_RL/chess_rl_policy.pth"
    ]
    
    head_type = 'linear'  # default fallback
    for path in policy_checkpoint_candidates:
        if os.path.exists(path):
            print(f"Attempting to load policy head from '{path}'...")
            try:
                policy_checkpoint = torch.load(path, map_location=device, weights_only=False)
                head_type = policy_checkpoint.get('head_type', 'linear')
                policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=True, head_type=head_type).to(device)
                policy_model.load_state_dict(policy_checkpoint['model_state_dict'])
                print(f"✅ Trained policy head loaded successfully from '{path}' (head_type='{head_type}').")
                loaded_policy = True
                break
            except Exception as e:
                print(f"⚠️ Could not load state dict from '{path}': {e}")
                
    if not loaded_policy:
        policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=True, head_type='linear').to(device)
        print("⚠️ Warning: No compatible trained policy head checkpoint loaded. Using randomly initialized policy head.")
        
    policy_model.eval()
    
    # 3. Define sample positions to evaluate
    sample_positions = [
        # Standard Initial Position
        {
            "name": "Standard Starting Board",
            "fen": chess.STARTING_FEN
        },
        # Tactical Middle-game Position (White to move, strong tactical options)
        {
            "name": "Tactical Middle-Game (White to play)",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        },
        # Checkmate in 1 (Scholar's Mate final position, White to deliver mate)
        {
            "name": "Scholar's Mate Setup (White to play Qh7#)",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        }
    ]
    
    # 4. Propose moves for each position
    print("\n" + "="*60)
    print("CHESS MOVE PROPOSER DEMONSTRATION")
    print("="*60)
    
    for idx, pos in enumerate(sample_positions):
        print(f"\nPosition {idx + 1}: {pos['name']}")
        print(f"FEN: {pos['fen']}")
        board = chess.Board(pos["fen"])
        print(f"Active player: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        # Propose using direct Policy Head
        best_policy_move = policy_model.propose_move(board, method='policy_head', device=device)
        # Propose using JEPA Predictor + Value Search
        best_predictor_move = policy_model.propose_move(board, method='jepa_predictor', device=device)
        
        print(f"👉 Proposed move (Policy Head):  {best_policy_move}")
        print(f"👉 Proposed move (JEPA Predictor): {best_predictor_move}")
        
        # Display Top 5 moves by Policy Probability
        print("\nTop 5 moves according to Policy Head (probabilities):")
        top_probs = policy_model.get_move_probabilities(board, device=device)[:5]
        for rank, (move, prob) in enumerate(top_probs):
            print(f"  {rank+1}. {move} | Probability: {prob:.4f}")
            
        # Display Top 5 moves by JEPA Predictor Value Search
        print("\nTop 5 moves according to JEPA Predictor + Value Head:")
        top_vals = policy_model.get_move_values(board, device=device)[:5]
        for rank, (move, val) in enumerate(top_vals):
            print(f"  {rank+1}. {move} | Predicted Value: {val:+.4f}")
            
        print("-"*60)

if __name__ == "__main__":
    main()
