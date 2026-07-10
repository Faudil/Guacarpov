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
                
                # Clean up memory explicitly
                del boards, next_boards, outcomes, chunk
                import gc
                gc.collect()
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
            print(f"Warning: metadata.pt not found. Estimating total samples from {len(self.chunk_files)} chunks...")
            self.total_samples = len(self.chunk_files) * 50_000 * 60

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
                
                # Clean up memory explicitly
                del all_states, lengths, outcomes, data
                del game_starts, splits, game_start_indices, starts, game_starts_array, is_last_state, valid_indices
                import gc
                gc.collect()
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
    
    enc_trainable = sum(p.numel() for p in model.context_encoder.parameters() if p.requires_grad) if hasattr(model, 'context_encoder') else 0
    enc_total = sum(p.numel() for p in model.context_encoder.parameters()) if hasattr(model, 'context_encoder') else 0
    pred_trainable = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad) if hasattr(model, 'predictor') else 0
    pred_total = sum(p.numel() for p in model.predictor.parameters()) if hasattr(model, 'predictor') else 0
    val_trainable = sum(p.numel() for p in model.value_head.parameters() if p.requires_grad) if hasattr(model, 'value_head') else 0
    val_total = sum(p.numel() for p in model.value_head.parameters()) if hasattr(model, 'value_head') else 0
    target_trainable = sum(p.numel() for p in model.target_encoder.parameters() if p.requires_grad) if hasattr(model, 'target_encoder') else 0
    target_total = sum(p.numel() for p in model.target_encoder.parameters()) if hasattr(model, 'target_encoder') else 0
    
    print(f"Model parameters: {num_trainable:,} trainable | {num_total:,} total")
    print(f"  • Encoder      : {enc_trainable:,} trainable | {enc_total:,} total")
    print(f"  • Predictor    : {pred_trainable:,} trainable | {pred_total:,} total")
    print(f"  • Value Head   : {val_trainable:,} trainable | {val_total:,} total")
    if target_total > 0:
        print(f"  • Target Enc   : {target_trainable:,} trainable | {target_total:,} total (EMA updated)")
    
    return train_loader, dataset, model, steps_per_epoch

def _train_jepa_step(model, optimizer, scaler, scheduler, boards, next_boards, actions, outcomes, use_amp, args, device, epoch=0, i=0, steps_per_epoch=1, num_epochs=1):
    boards = boards.to(device)
    next_boards = next_boards.to(device)
    actions = actions.to(device)
    outcomes = outcomes.to(device).float().view(-1, 1)
    
    if args.arch == 'spatiotemporal':
        boards = boards.unsqueeze(1)
        next_boards = next_boards.unsqueeze(1)
        
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
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
        
        # Add Covariance Loss (VICReg) to prevent dimensional collapse
        B_size, D_size = context_latents.shape
        # Context Covariance
        ctx_centered = context_latents - context_latents.mean(dim=0)
        cov_ctx = (ctx_centered.T @ ctx_centered) / (B_size - 1)
        cov_loss_ctx = (cov_ctx.pow(2).sum() - cov_ctx.diag().pow(2).sum()) / D_size
        
        # Predictor Covariance
        pred_flat = pred_target_latents.view(B_size, -1)
        pred_centered = pred_flat - pred_flat.mean(dim=0)
        cov_pred = (pred_centered.T @ pred_centered) / (B_size - 1)
        cov_loss_pred = (cov_pred.pow(2).sum() - cov_pred.diag().pow(2).sum()) / pred_flat.size(1)
        
        cov_loss = cov_loss_ctx + cov_loss_pred
        
        # Standard VICReg weighting usually puts high weight on cov/var
        total_loss = jepa_loss + args.val_coeff * val_loss + var_loss + cov_loss
        
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
        
    scheduler.step()
        
    global_step = epoch * steps_per_epoch + i
    total_steps = num_epochs * steps_per_epoch
    # Cosine annealing from args.ema_decay to 1.0
    current_decay = 1.0 - (1.0 - args.ema_decay) * (math.cos(math.pi * global_step / total_steps) + 1.0) / 2.0
        
    model.update_target_encoder(decay=current_decay)
    return jepa_loss.item(), val_loss.item(), var_loss.item(), cov_loss.item(), total_loss.item()

def _train_jepa_epoch(model, train_loader, optimizer, scaler, scheduler, use_amp, steps_per_epoch, epoch, num_epochs, args, device):
    model.train()
    running_jepa_loss = 0.0
    running_val_loss = 0.0
    running_var_loss = 0.0
    running_cov_loss = 0.0
    running_total_loss = 0.0
    
    pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=30.0)
    for i, (boards, next_boards, actions, outcomes) in enumerate(pbar):
        if i >= steps_per_epoch:
            break
            
        jepa_l, val_l, var_l, cov_l, tot_l = _train_jepa_step(
            model, optimizer, scaler, scheduler, boards, next_boards, actions, outcomes, use_amp, args, device,
            epoch=epoch, i=i, steps_per_epoch=steps_per_epoch, num_epochs=num_epochs
        )
        
        if math.isnan(tot_l):
            print(f"\nCRITICAL: NaN loss detected at step {i}! Halting epoch to prevent model corruption.")
            pbar.close()
            return False
        
        running_jepa_loss += jepa_l
        running_val_loss += val_l
        running_var_loss += var_l
        running_cov_loss += cov_l
        running_total_loss += tot_l
        
        pbar.set_postfix(Total=f"{tot_l:.4f}", Latent_MSE=f"{jepa_l:.4f}", Var_Pen=f"{var_l:.4f}", Cov_Pen=f"{cov_l:.4f}", Value=f"{val_l:.4f}")
    pbar.close()
    
    n = max(steps_per_epoch, 1)
    avg_jepa = running_jepa_loss / n
    avg_var = running_var_loss / n
    avg_cov = running_cov_loss / n
    avg_val = running_val_loss / n
    avg_tot = running_total_loss / n
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  • Total Loss : {avg_tot:.4f} (Overall objective, ↓ lower is better)")
    print(f"  • Latent MSE : {avg_jepa:.4f} (Predicting next state, ↓ lower is better, ideal ~0.05-0.2)")
    print(f"  • Var Penalty: {avg_var:.4f} (Ensures variance, ↓ lower is better, ideal 0.0)")
    print(f"  • Cov Penalty: {avg_cov:.4f} (Prevents redundancy, ↓ lower is better, ideal 0.0)")
    print(f"  • Value Loss : {avg_val:.4f} (Predicting outcome, ↓ lower is better)")
    
    # Threshold checks for representation collapse
    if avg_jepa < 1e-3:
        print("  ⚠️  WARNING: Latent MSE is suspiciously low (< 0.001). The model may have collapsed to a trivial constant representation.")
    elif avg_jepa > 2.0:
        print("  ⚠️  WARNING: Latent MSE is very high. The predictor is struggling to learn the latent dynamics.")
        
    if avg_var > 1.2:
        print("  ⚠️  WARNING: Variance Penalty is high (> 1.2). Dimensional collapse may be occurring (embeddings lack diversity).")
        
    # SFT Usability Check
    is_usable = True
    reasons = []
    
    if avg_jepa < 1e-3:
        is_usable = False
        reasons.append("Latent MSE too low (Collapse)")
    elif avg_jepa > 0.5:
        is_usable = False
        reasons.append("Latent MSE too high (Poor Dynamics)")
        
    if avg_var > 0.8:
        is_usable = False
        reasons.append("High Var Penalty (Low Diversity)")
        
    if avg_cov > 0.5:
        is_usable = False
        reasons.append("High Cov Penalty (High Redundancy)")

    print("  --------------------------------------------------")
    if is_usable:
        print("  ✅ SFT Readiness: USABLE. Representations are healthy and ready for policy fine-tuning.")
    else:
        print("  ❌ SFT Readiness: NOT USABLE. Issues: " + " | ".join(reasons))
        
    print()
    return True

def train_jepa(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, dataset, model, steps_per_epoch = _load_jepa_dataset_and_model(args, device)
    
    enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    pred_lr = args.predictor_lr if args.predictor_lr is not None else args.lr
    enc_wd = args.encoder_weight_decay if args.encoder_weight_decay is not None else args.weight_decay
    pred_wd = args.predictor_weight_decay if args.predictor_weight_decay is not None else args.weight_decay

    trainable_params = [
        {'params': model.context_encoder.parameters(), 'lr': enc_lr, 'weight_decay': enc_wd},
        {'params': model.predictor.parameters(), 'lr': pred_lr, 'weight_decay': pred_wd},
        {'params': model.value_head.parameters(), 'lr': enc_lr, 'weight_decay': enc_wd},
    ]
    optimizer = optim.AdamW(trainable_params)
    
    total_steps = args.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[enc_lr, pred_lr, enc_lr], 
        total_steps=total_steps,
        pct_start=0.1, # 10% warmup to prevent early divergence
        anneal_strategy='cos'
    )
    
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=use_amp)
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nLoading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                steps_to_catch_up = start_epoch * steps_per_epoch
                if steps_to_catch_up > 0:
                    print(f"Advancing LR scheduler by {steps_to_catch_up} steps to catch up...")
                    for _ in range(steps_to_catch_up):
                        scheduler.step()
                        
            if 'scaler_state_dict' in checkpoint and use_amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        
            print(f"Successfully resumed from epoch {start_epoch} (Next epoch to train: {start_epoch+1})")
        else:
            print(f"\nWarning: Checkpoint '{args.resume}' not found. Starting from scratch.")

    print("\nStarting JEPA training with AMP enabled...\n") if use_amp else print("\nStarting JEPA training...\n")
    for epoch in range(start_epoch, args.epochs):
        success = _train_jepa_epoch(model, train_loader, optimizer, scaler, scheduler, use_amp, steps_per_epoch, epoch, args.epochs, args, device)
        
        os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
        
        # Overwrite latest checkpoint
        save_dict = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'args': args
        }
        torch.save(save_dict, args.save_path)
        
        # Save a unique checkpoint for this epoch
        base, ext = os.path.splitext(args.save_path)
        epoch_path = f"{base}_ep{epoch+1}{ext}"
        torch.save(save_dict, epoch_path)
        
        print(f"Checkpoints saved to '{args.save_path}' and '{epoch_path}'")
        
        if not success:
            print("\nTraining aborted early to preserve the previous epoch's valid checkpoint!")
            break
            
    print("\n🎉 JEPA v2 training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chess JEPA v3.")
    parser.add_argument('--data_dir', type=str, default='jepa_v3_data', help="Path to processed data chunks.")
    parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'convnext', 'vit', 'spatiotemporal'], help="JEPA architecture to use.")
    parser.add_argument('--in_channels', type=int, default=111, help="Number of input channels (12 for v2 data, 111 for v3).")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size.") 
    parser.add_argument('--num_workers', type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Default learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Default weight decay.")
    parser.add_argument('--encoder_lr', type=float, default=None, help="Learning rate for the encoder. Defaults to --lr if not set.")
    parser.add_argument('--predictor_lr', type=float, default=None, help="Learning rate for the predictor. Defaults to --lr if not set.")
    parser.add_argument('--encoder_weight_decay', type=float, default=None, help="Weight decay for the encoder. Defaults to --weight_decay if not set.")
    parser.add_argument('--predictor_weight_decay', type=float, default=None, help="Weight decay for the predictor. Defaults to --weight_decay if not set.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs.")
    parser.add_argument('--ema_decay', type=float, default=0.996, help="EMA decay for target encoder.")
    parser.add_argument('--val_coeff', type=float, default=0.5, help="Weight for value/outcome loss.")
    parser.add_argument('--latent_dim', type=int, default=256, help="Latent dimension (d_model).")
    parser.add_argument('--num_res_blocks', type=int, default=8, help="Residual blocks in encoder (Legacy ResNet).")
    parser.add_argument('--num_filters', type=int, default=128, help="Conv filters in encoder (Legacy ResNet).")
    parser.add_argument('--spatial_dim', type=int, default=64, help="ConvNeXt filter dimension for Spatiotemporal.")
    parser.add_argument('--spatial_blocks', type=int, default=16, help="ConvNeXt block count for Spatiotemporal.")
    parser.add_argument('--temporal_layers', type=int, default=8, help="Transformer layer count for Spatiotemporal.")
    parser.add_argument('--predictor_layers', type=int, default=None, help="Number of Transformer Decoder layers in the Predictor. Defaults to --temporal_layers if not set.")
    parser.add_argument('--temporal_heads', type=int, default=16, help="Transformer head count for Spatiotemporal.")
    parser.add_argument('--save_path', type=str, default='JEPA/chess_jepa.pth', help="Checkpoint save path.")
    parser.add_argument('--num_games', type=int, default=None, help="Number of games to limit training to (approximate).")
    parser.add_argument('--action_conditioned', action='store_true', help="Condition the predictor on the played move (World Model mode).")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume training from (e.g. JEPA/chess_jepa_ep1.pth).")
    
    args = parser.parse_args()
    train_jepa(args)
