import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import chess
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.dirname(__file__))

from policy_model import ChessJepaPolicy, board_to_tensor

class PolicyDistillDataset(IterableDataset):
    """
    An iterable dataset that reads raw FENs from the CSV,
    uses the pre-trained JEPA model to evaluate all legal moves in a single batch on the GPU,
    and yields (board_tensor, target_policy_distribution) for distillation training.
    """
    def __init__(self, csv_path, jepa_model, move_mapper, limit=50000, device='cpu', temp=0.1):
        self.csv_path = csv_path
        self.jepa_model = jepa_model
        self.move_mapper = move_mapper
        self.limit = limit
        self.device = device
        self.temp = temp
        
    def _evaluate_moves(self, board, legal_moves):
        with torch.no_grad():
            legal_indices = []
            pred_vals_list = []
            for move in legal_moves:
                legal_indices.append(self.move_mapper.move_to_index(move))
                next_board = board.copy()
                next_board.push(move)
                next_board_t = board_to_tensor(next_board).unsqueeze(0).to(self.device)
                next_latent = self.jepa_model.forward_context(next_board_t)
                val = self.jepa_model.predict_value(next_latent)[0, 0]
                pred_vals_list.append(val)
            pred_vals = torch.stack(pred_vals_list)
        return legal_indices, pred_vals

    def __iter__(self):
        df_iter = pd.read_csv(self.csv_path, chunksize=1000)
        count = 0
        self.jepa_model.eval()
        for chunk in df_iter:
            if count >= self.limit:
                break
            for _, row in chunk.iterrows():
                if count >= self.limit:
                    break
                fen = row['FEN']
                try:
                    board = chess.Board(fen)
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        continue
                        
                    board_t = board_to_tensor(board).to(self.device)
                    legal_indices, pred_vals = self._evaluate_moves(board, legal_moves)
                    
                    target_probs = torch.zeros(4672, dtype=torch.float32)
                    multiplier = 1.0 if board.turn == chess.WHITE else -1.0
                    scaled_vals = pred_vals * (multiplier / self.temp)
                    probs = F.softmax(scaled_vals, dim=0)
                    target_probs[legal_indices] = probs.cpu()
                    
                    yield board_t.cpu(), target_probs
                    count += 1
                except Exception:
                    continue

def _load_distill_components(args, device):
    if not os.path.exists(args.jepa_checkpoint):
        raise FileNotFoundError(f"JEPA checkpoint not found at {args.jepa_checkpoint}. Train JEPA first!")
        
    from factory import load_jepa_from_checkpoint
    jepa_model, jepa_args = load_jepa_from_checkpoint(args.jepa_checkpoint, device)
    print("✅ Pre-trained JEPA model loaded successfully.")
    
    policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=True, head_type=args.head_type).to(device)
    head_params = sum(p.numel() for p in policy_model.policy_head.parameters())
    print(f"Policy head: '{args.head_type}' ({head_params:,} params)")
    
    print("Initializing policy distillation dataset (evaluating moves on the fly)...")
    dataset = PolicyDistillDataset(
        csv_path=args.csv_path,
        jepa_model=jepa_model,
        move_mapper=policy_model.move_mapper,
        limit=args.limit,
        device=device,
        temp=args.temperature
    )
    return policy_model, dataset, jepa_args

def _train_distill_epoch(policy_model, train_loader, optimizer, steps_per_epoch, device, epoch, num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}")
    policy_model.train()
    
    for i, (boards, target_probs) in enumerate(progress_bar):
        boards = boards.to(device)
        target_probs = target_probs.to(device)
        
        logits = policy_model(boards)
        mask = torch.full_like(logits, float('-inf'))
        mask[target_probs > 0] = 0.0
        masked_logits = logits + mask
        
        log_probs = F.log_softmax(masked_logits, dim=-1)
        product = target_probs * log_probs
        product = torch.where(target_probs > 0, product, torch.zeros_like(product))
        loss = -torch.sum(product, dim=-1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    progress_bar.close()
    return running_loss / steps_per_epoch

def train_policy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_model, dataset, jepa_args = _load_distill_components(args, device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False)
    
    optimizer = optim.AdamW(policy_model.policy_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(args.limit / args.batch_size))
    print(f"Distillation steps per epoch: {steps_per_epoch}")
    
    print("\nStarting distillation training...")
    for epoch in range(args.epochs):
        epoch_loss = _train_distill_epoch(policy_model, train_loader, optimizer, steps_per_epoch, device, epoch, args.epochs)
        print(f"Epoch {epoch+1} Distillation Loss: {epoch_loss:.4f}")
        
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': policy_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss, 'jepa_args': jepa_args, 'head_type': args.head_type, 'args': args
        }, args.save_path)
        print(f"Policy checkpoint saved to '{args.save_path}'")
    print("\n🎉 Policy distillation training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chess JEPA Policy Head.")
    parser.add_argument('--csv_path', type=str, default='chess-evaluations/chessData.csv', help="Path to raw chessData.csv.")
    parser.add_argument('--jepa_checkpoint', type=str, default='JEPA/chess_jepa.pth', help="Path to pre-trained JEPA model.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for policy training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--limit', type=int, default=20000, help="Number of FEN samples to distill on per epoch.")
    parser.add_argument('--temperature', type=float, default=0.1, help="Softmax temperature for value targets.")
    parser.add_argument('--save_path', type=str, default='JEPA_Policy/chess_policy.pth', help="Path to save policy model.")
    parser.add_argument('--head_type', type=str, default='linear', choices=['linear', 'mlp', 'transformer', 'moe'],
                        help="Policy head architecture: linear (default), mlp, transformer, or moe.")
    args = parser.parse_args()
    train_policy(args)
