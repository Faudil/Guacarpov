import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import io
import os
import sys
import argparse
import random
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.dirname(__file__))

from policy_model import ChessJepaPolicy
from data_utils import build_111_batch

class ChessSftDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx].float(), self.actions[idx]

class ChessSftV3Dataset(Dataset):
    def __init__(self, states, actions, game_starts):
        self.states = states
        self.actions = actions
        self.game_starts = game_starts
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return idx, self.actions[idx]

def collate_fn_v3(batch, dataset):
    indices = np.array([item[0] for item in batch])
    actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
    boards = build_111_batch(indices, dataset.states, dataset.game_starts)
    return boards, actions

def _load_sft_dataset(args):
    if args.format == '12':
        cache_file = f"JEPA_Policy/sft_cache_format12_g{args.num_games}_elo{args.min_elo}.pt"
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache {cache_file} not found! Please run prepare_sft_dataset.py --format 12 first.")
        print(f"Loading cached 12-channel dataset from '{cache_file}'...")
        cached_data = torch.load(cache_file, map_location='cpu', weights_only=True)
        dataset = ChessSftDataset(cached_data['states'], cached_data['actions'])
        collate_fn = None
    else:
        cache_file = f"JEPA_Policy/sft_cache_format111_g{args.num_games}_elo{args.min_elo}.npz"
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache {cache_file} not found! Please run prepare_sft_dataset.py --format 111 first.")
        print(f"Loading cached 111-channel dataset from '{cache_file}'...")
        data = np.load(cache_file)
        dataset = ChessSftV3Dataset(data['states'], data['actions'], data['game_starts'])
        collate_fn = lambda batch: collate_fn_v3(batch, dataset)
        
    num_samples = len(dataset)
    print(f"✅ Loaded {num_samples} positions.")
    val_size = int(num_samples * args.val_split)
    train_size = num_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    print(f"Dataset split: Train samples = {train_size}, Validation samples = {val_size}")
    return train_loader, val_loader

def _load_sft_models_optimizer(args, device):
    if not os.path.exists(args.jepa_checkpoint):
        raise FileNotFoundError(f"JEPA checkpoint not found at '{args.jepa_checkpoint}'. Run JEPA training first!")
        
    from factory import load_jepa_from_checkpoint
    jepa_model, jepa_args = load_jepa_from_checkpoint(args.jepa_checkpoint, device)
    
    policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=args.freeze_backbone, head_type=args.head_type).to(device)
    head_params = sum(p.numel() for p in policy_model.policy_head.parameters())
    print(f"Policy head: '{args.head_type}' ({head_params:,} params)")
    
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    return policy_model, optimizer, criterion, jepa_args

def _train_sft_epoch(policy_model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    policy_model.train()
    total_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", mininterval=30.0)
    for batch_states, batch_actions in train_pbar:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        
        logits = policy_model(batch_states)
        loss = criterion(logits, batch_actions)
        
        if hasattr(policy_model.policy_head, 'aux_loss'):
            loss = loss + policy_model.policy_head.aux_loss
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_pbar.set_postfix(loss=loss.item())
    train_pbar.close()
    return total_loss / len(train_loader)

def _validate_sft(policy_model, val_loader, criterion, device):
    policy_model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_states, batch_actions in val_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            logits = policy_model(batch_states)
            loss = criterion(logits, batch_actions)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == batch_actions).sum().item()
            total_predictions += batch_actions.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_val_loss, val_accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = _load_sft_dataset(args)
    policy_model, optimizer, criterion, jepa_args = _load_sft_models_optimizer(args, device)
    
    best_val_acc = 0.0
    print("\nStarting SFT Fine-Tuning...")
    for epoch in range(args.epochs):
        if args.freeze_backbone and args.unfreeze_epoch > 0 and (epoch + 1) == args.unfreeze_epoch:
            print(f"\n[Stage 2] Epoch {epoch+1}: Unfreezing JEPA backbone for end-to-end training...")
            for param in policy_model.jepa_model.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW(policy_model.parameters(), lr=args.unfreeze_lr, weight_decay=args.weight_decay)
            print(f"Re-initialized optimizer with LR={args.unfreeze_lr} for all parameters.")

        avg_train_loss = _train_sft_epoch(policy_model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        avg_val_loss, val_accuracy = _validate_sft(policy_model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Top-1 Accuracy: {val_accuracy*100:.2f}%")
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': policy_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'jepa_args': jepa_args, 'head_type': args.head_type, 'val_accuracy': val_accuracy, 'args': args
            }, args.save_path)
            print(f"⭐ New best checkpoint saved to '{args.save_path}' (Val Acc: {val_accuracy*100:.2f}%)")
    print(f"\nSupervised Fine-Tuning complete! Best Val Accuracy: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune Chess Policy via Supervised SFT on Lichess.")
    parser.add_argument('--jepa_checkpoint', type=str, default='JEPA/chess_jepa_ep1.pth', help="Path to pre-trained JEPA model.")
    parser.add_argument('--save_path', type=str, default='JEPA_Policy/chess_sft_policy.pth', help="Path to save SFT checkpoint.")
    parser.add_argument('--num_games', type=int, default=100_000, help="Number of games to parse from HF dataset.")
    parser.add_argument('--min_elo', type=int, default=2200, help="Minimum rating for both players to include the game.")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for SFT.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of SFT training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay.")
    parser.add_argument('--val_split', type=float, default=0.1, help="Validation set split ratio.")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze JEPA context encoder backbone.")
    parser.add_argument('--unfreeze_epoch', type=int, default=-1, help="Epoch to unfreeze backbone (Two-Stage Training). -1 to disable.")
    parser.add_argument('--unfreeze_lr', type=float, default=1e-5, help="Learning rate after unfreezing backbone.")
    parser.add_argument('--no_cache', action='store_true', help="Disable dataset caching.")
    parser.add_argument('--head_type', type=str, default='linear', choices=['linear', 'mlp', 'transformer', 'moe', 'latent_thinker', 'moe_latent_thinker'],
                        help="Policy head architecture: linear (default), mlp, transformer, moe, latent_thinker, or moe_latent_thinker.")
    parser.add_argument('--format', type=str, choices=['12', '111'], default='111', help="Format of the dataset to load.")
    args = parser.parse_args()
    main(args)
