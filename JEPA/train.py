"""
JEPA v2 Training Loop

Training objective:
  L = MSE(predictor(context_enc(board_t)), target_enc(board_{t+1})) + λ * MSE(value_head(context_enc(board_t)), outcome)

Collapse prevention:
  - Target encoder: EMA of context encoder (standard JEPA)
  - BYOL-style: predictor is asymmetric (smaller), target is stop-gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import os
import glob
import numpy as np
import math
from tqdm import tqdm
import argparse

from jepa_model import ChessJEPA
from jepa_convnext import ChessJEPA_ConvNeXt
from jepa_vit import ChessJEPA_ViT
from jepa_spatiotemporal import ChessJEPA_SpatioTemporal
from data_utils import build_111_batch

class ChessJepaV2Dataset(IterableDataset):
    """
    Iterable dataset that loads pre-processed JEPA v2 transition chunks.
    Yields (board_t, board_{t+1}, outcome).
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, 'chunk_*.pt')))
        
        if not self.chunk_files:
            raise FileNotFoundError(f"No data chunks found in '{data_dir}'. Run prepare_jepa_v2.py first!")
        
        try:
            metadata = torch.load(os.path.join(data_dir, 'metadata.pt'), weights_only=True)
            self.total_samples = metadata['total_transitions']
        except Exception:
            print("Warning: metadata.pt not found. Estimating total samples...")
            self.total_samples = sum(
                len(torch.load(f, weights_only=True)['boards']) for f in self.chunk_files
            )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunks = self.chunk_files if worker_info is None else self.chunk_files[worker_info.id::worker_info.num_workers]
        np.random.shuffle(chunks)
        
        for chunk_file in chunks:
            try:
                chunk = torch.load(chunk_file, weights_only=True)
                boards = chunk['boards']
                next_boards = chunk['next_boards']
                outcomes = chunk['outcomes']
                
                indices = np.arange(len(boards))
                np.random.shuffle(indices)
                for idx in indices:
                    yield (boards[idx].float(), next_boards[idx].float(), torch.tensor(-1, dtype=torch.long), outcomes[idx])
            except Exception as e:
                print(f"Error reading chunk {chunk_file}: {e}")
                continue

    def __len__(self):
        return self.total_samples

class ChessJepaV3Dataset(IterableDataset):
    """
    Iterable dataset that loads pre-processed JEPA v3 trajectory chunks (.npz)
    and dynamically reconstructs 111-channel AlphaZero style tensors.
    """
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, 'chunk_*.npz')))
        
        if not self.chunk_files:
            raise FileNotFoundError(f"No .npz data chunks found in '{data_dir}'. Run prepare_jepa_v3.py first!")
        
        try:
            metadata = torch.load(os.path.join(data_dir, 'metadata.pt'), weights_only=True)
            self.total_samples = metadata.get('games_processed', len(self.chunk_files)*50000) * 60
        except Exception:
            self.total_samples = 1_000_000

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunks = self.chunk_files if worker_info is None else self.chunk_files[worker_info.id::worker_info.num_workers]
        np.random.shuffle(chunks)
        
        for chunk_file in chunks:
            try:
                data = np.load(chunk_file)
                all_states = data['states']
                lengths = data['lengths']
                outcomes = data['outcomes']
                N = len(all_states)
                
                game_starts = np.zeros(N, dtype=np.int32)
                splits = np.cumsum(lengths)[:-1]
                if len(splits) > 0:
                    game_starts[splits] = 1
                game_start_indices = np.cumsum(game_starts)
                
                starts = np.insert(splits, 0, 0)
                game_starts_array = starts[game_start_indices]
                
                is_last_state = np.zeros(N, dtype=bool)
                is_last_state[starts + lengths - 1] = True
                
                valid_indices = np.where(~is_last_state)[0]
                np.random.shuffle(valid_indices)
                
                for b in range(0, len(valid_indices), self.batch_size):
                    batch_idx = valid_indices[b:b+self.batch_size]
                    if len(batch_idx) == 0: continue
                    
                    b_boards = build_111_batch(batch_idx, all_states, game_starts_array)
                    b_next_boards = build_111_batch(batch_idx + 1, all_states, game_starts_array)
                    
                    if 'action' in all_states.dtype.names:
                        b_actions = torch.from_numpy(all_states['action'][batch_idx]).long()
                    else:
                        b_actions = torch.full((len(batch_idx),), -1, dtype=torch.long)
                        
                    b_outcomes = torch.from_numpy(outcomes[game_start_indices[batch_idx]]).float()
                    
                    yield (b_boards, b_next_boards, b_actions, b_outcomes)
            except Exception as e:
                print(f"Error reading chunk {chunk_file}: {e}")
                continue

    def __len__(self):
        return self.total_samples

def _load_jepa_dataset_and_model(args, device):
    has_npz = len(glob.glob(os.path.join(args.data_dir, '*.npz'))) > 0
    if has_npz:
        print(f"Loading JEPA v3 dataset (npz) from '{args.data_dir}'...")
        dataset = ChessJepaV3Dataset(args.data_dir, batch_size=args.batch_size)
        loader_batch_size = None
    else:
        print(f"Loading JEPA v2 dataset (pt) from '{args.data_dir}'...")
        dataset = ChessJepaV2Dataset(args.data_dir)
        loader_batch_size = args.batch_size
        
    train_loader = DataLoader(dataset, batch_size=loader_batch_size, num_workers=args.num_workers, pin_memory=True)
    
    if args.num_games is not None:
        max_samples = args.num_games * 60
        dataset.total_samples = min(dataset.total_samples, max_samples)
        
    steps_per_epoch = math.ceil(dataset.total_samples / args.batch_size)
    print(f"Total transitions: {dataset.total_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    print(f"Initializing {args.arch} JEPA v2 model...")
    from factory import build_jepa_model
    model = build_jepa_model(args).to(device)
    
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_trainable:,} trainable | {num_total:,} total")
    
    return train_loader, dataset, model, steps_per_epoch

def _train_jepa_step(model, optimizer, scaler, boards, next_boards, actions, outcomes, use_amp, args, device):
    boards = boards.to(device)
    next_boards = next_boards.to(device)
    actions = actions.to(device)
    outcomes = outcomes.to(device).float().view(-1, 1)
    
    if args.arch == 'spatiotemporal':
        boards = boards.unsqueeze(1)
        next_boards = next_boards.unsqueeze(1)
        
    with torch.autocast(device_type=device.type, enabled=use_amp):
        context_latents = model.forward_context(boards)
        
        if getattr(args, 'action_conditioned', False):
            pred_target_latents = model.forward_predict(context_latents, actions)
        else:
            pred_target_latents = model.forward_predict(context_latents)
            
        target_latents = model.forward_target(next_boards)
        pred_outcome = model.predict_value(context_latents)
        
        jepa_loss = F.mse_loss(pred_target_latents, target_latents.detach())
        val_loss = F.smooth_l1_loss(pred_outcome, outcomes)
        
        std_ctx = torch.sqrt(context_latents.var(dim=0) + 1e-4)
        std_pred = torch.sqrt(pred_target_latents.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std_ctx)) + torch.mean(F.relu(1.0 - std_pred))
        total_loss = jepa_loss + args.val_coeff * val_loss + var_loss
        
    optimizer.zero_grad()
    if use_amp:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    model.update_target_encoder(decay=args.ema_decay)
    return jepa_loss.item(), val_loss.item(), var_loss.item(), total_loss.item()

def _train_jepa_epoch(model, train_loader, optimizer, scaler, use_amp, steps_per_epoch, epoch, num_epochs, args, device):
    model.train()
    running_jepa_loss = 0.0
    running_val_loss = 0.0
    running_var_loss = 0.0
    running_total_loss = 0.0
    
    pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=30.0)
    for i, (boards, next_boards, actions, outcomes) in enumerate(pbar):
        if i >= steps_per_epoch:
            break
            
        jepa_l, val_l, var_l, tot_l = _train_jepa_step(model, optimizer, scaler, boards, next_boards, actions, outcomes, use_amp, args, device)
        
        running_jepa_loss += jepa_l
        running_val_loss += val_l
        running_var_loss += var_l
        running_total_loss += tot_l
        
        pbar.set_postfix(total=f"{tot_l:.4f}", jepa=f"{jepa_l:.4f}", var=f"{var_l:.4f}", val=f"{val_l:.4f}")
    pbar.close()
    
    n = max(steps_per_epoch, 1)
    print(f"Epoch {epoch+1} — Total: {running_total_loss/n:.4f} | JEPA: {running_jepa_loss/n:.4f} | Var: {running_var_loss/n:.4f} | Value: {running_val_loss/n:.4f}")

def train_jepa(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, dataset, model, steps_per_epoch = _load_jepa_dataset_and_model(args, device)
    
    trainable_params = [
        {'params': model.context_encoder.parameters(), 'lr': args.lr},
        {'params': model.predictor.parameters(), 'lr': args.lr},
        {'params': model.value_head.parameters(), 'lr': args.lr},
    ]
    optimizer = optim.AdamW(trainable_params, weight_decay=args.weight_decay)
    
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=use_amp)
    
    print("\nStarting JEPA training with AMP enabled...\n") if use_amp else print("\nStarting JEPA training...\n")
    for epoch in range(args.epochs):
        _train_jepa_epoch(model, train_loader, optimizer, scaler, use_amp, steps_per_epoch, epoch, args.epochs, args, device)
        
        os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'args': args
        }, args.save_path)
        print(f"Checkpoint saved to '{args.save_path}'")
    print("\n🎉 JEPA v2 training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chess JEPA v3.")
    parser.add_argument('--data_dir', type=str, default='jepa_v3_data', help="Path to processed data chunks.")
    parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'convnext', 'vit', 'spatiotemporal'], help="JEPA architecture to use.")
    parser.add_argument('--in_channels', type=int, default=111, help="Number of input channels (12 for v2 data, 111 for v3).")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size.") 
    parser.add_argument('--num_workers', type=int, default=12, help="Number of dataloader workers.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs.")
    parser.add_argument('--ema_decay', type=float, default=0.996, help="EMA decay for target encoder.")
    parser.add_argument('--val_coeff', type=float, default=0.5, help="Weight for value/outcome loss.")
    parser.add_argument('--latent_dim', type=int, default=256, help="Latent dimension (d_model).")
    parser.add_argument('--num_res_blocks', type=int, default=8, help="Residual blocks in encoder (Legacy ResNet).")
    parser.add_argument('--num_filters', type=int, default=128, help="Conv filters in encoder (Legacy ResNet).")
    parser.add_argument('--spatial_dim', type=int, default=64, help="ConvNeXt filter dimension for Spatiotemporal.")
    parser.add_argument('--spatial_blocks', type=int, default=16, help="ConvNeXt block count for Spatiotemporal.")
    parser.add_argument('--temporal_layers', type=int, default=8, help="Transformer layer count for Spatiotemporal.")
    parser.add_argument('--temporal_heads', type=int, default=16, help="Transformer head count for Spatiotemporal.")
    parser.add_argument('--save_path', type=str, default='JEPA/chess_jepa.pth', help="Checkpoint save path.")
    parser.add_argument('--num_games', type=int, default=None, help="Number of games to limit training to (approximate).")
    parser.add_argument('--action_conditioned', action='store_true', help="Condition the predictor on the played move (World Model mode).")
    
    args = parser.parse_args()
    train_jepa(args)
