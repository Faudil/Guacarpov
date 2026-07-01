import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import os
import sys
import argparse
import multiprocessing as mp
from tqdm import tqdm

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA_Policy"))

from policy_model import ChessJepaPolicy
from self_play import self_play_worker

class ChessRlDataset(Dataset):
    def __init__(self, states, actions, legal_masks, advantages, targets, old_log_probs):
        self.states = states
        self.actions = actions
        self.legal_masks = legal_masks
        self.advantages = advantages
        self.targets = targets
        self.old_log_probs = old_log_probs
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return (
            self.states[idx].float(),
            self.actions[idx],
            self.legal_masks[idx],
            self.advantages[idx],
            self.targets[idx],
            self.old_log_probs[idx]
        )

def _load_rl_models(args, device):
    from factory import build_jepa_model
    policy_model = None
    loaded_policy = False
    
    policy_candidates = []
    if args.policy_checkpoint:
        policy_candidates.append(args.policy_checkpoint)
    for path in ['JEPA_Policy/chess_sft_policy.pth', 'JEPA_RL/chess_rl_policy.pth', 'JEPA_Policy/chess_policy.pth']:
        if path not in policy_candidates:
            policy_candidates.append(path)
            
    for path in policy_candidates:
        if os.path.exists(path):
            print(f"Attempting to load base policy model from '{path}'...")
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                jepa_args = checkpoint['jepa_args'] if 'jepa_args' in checkpoint else checkpoint['args']
                jepa_model = build_jepa_model(jepa_args).to(device)
                ckpt_head_type = checkpoint.get('head_type', args.head_type)
                policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=args.freeze_backbone, head_type=ckpt_head_type).to(device)
                policy_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded base policy model successfully from '{path}'.")
                loaded_policy = True
                break
            except Exception as e:
                print(f"⚠️ Could not load policy checkpoint from '{path}': {e}")
                
    if not loaded_policy and os.path.exists(args.jepa_checkpoint):
        print(f"No compatible policy checkpoint found. Creating a new policy head on top of pre-trained JEPA model '{args.jepa_checkpoint}'...")
        try:
            checkpoint = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
            jepa_args = checkpoint['args']
            jepa_model = build_jepa_model(jepa_args).to(device)
            jepa_model.load_state_dict(checkpoint['model_state_dict'])
            policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=args.freeze_backbone, head_type=args.head_type).to(device)
            loaded_policy = True
            print("✅ Pre-trained JEPA model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading JEPA model from '{args.jepa_checkpoint}': {e}")
            
    if not loaded_policy:
        raise FileNotFoundError("No compatible JEPA or Policy checkpoints could be loaded!")
        
    import copy
    ref_policy_model = copy.deepcopy(policy_model)
    ref_policy_model.eval()
    ref_policy_model.requires_grad_(False)
    return policy_model, ref_policy_model, jepa_args

def _collect_multiprocess_self_play(args, temp_checkpoint_path, device):
    games_per_worker = args.games_per_epoch // args.num_selfplay_workers
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=args.num_selfplay_workers)
    async_results = []
    opp_probs = {'sft': args.opp_sft, 'random': args.opp_random, 'stockfish': args.opp_stockfish, 'self': args.opp_self}
    
    for w_id in range(args.num_selfplay_workers):
        res = pool.apply_async(
            self_play_worker,
            args=(w_id, games_per_worker, args.temperature, args.max_moves, temp_checkpoint_path, str(device), args.gamma, args.policy_checkpoint, args.stockfish_path, opp_probs)
        )
        async_results.append(res)
    pool.close()
    pool.join()
    
    worker_states, worker_actions, worker_turns, worker_legal_masks, worker_targets, worker_old_log_probs, all_outcomes = [], [], [], [], [], [], []
    for res in async_results:
        worker_data = res.get()
        if worker_data is None: continue
        states_arr, actions_arr, turns_arr, legal_masks_arr, targets_arr, old_lp_arr, outcomes = worker_data
        worker_states.append(states_arr)
        worker_actions.append(actions_arr)
        worker_turns.append(turns_arr)
        worker_legal_masks.append(legal_masks_arr)
        worker_targets.append(targets_arr)
        worker_old_log_probs.append(old_lp_arr)
        all_outcomes.extend(outcomes)
        
    if not worker_states:
        return None
        
    states_np = np.concatenate(worker_states, axis=0)
    draws = all_outcomes.count(0.0)
    white_wins = all_outcomes.count(1.0)
    black_wins = all_outcomes.count(-1.0)
    avg_length = len(states_np) / max(1, len(all_outcomes))
    
    return (
        states_np,
        np.concatenate(worker_actions, axis=0),
        np.concatenate(worker_turns, axis=0),
        np.concatenate(worker_legal_masks, axis=0),
        np.concatenate(worker_targets, axis=0),
        np.concatenate(worker_old_log_probs, axis=0),
        (avg_length, white_wins, black_wins, draws)
    )

def _compute_advantages(policy_model, states_tensor, turns_np, targets_tensor, device):
    all_values = []
    policy_model.eval()
    with torch.no_grad():
        for i in range(0, len(states_tensor), 4096):
            chunk_states = states_tensor[i:i+4096].to(device).float()
            latent = policy_model.jepa_model.forward_context(chunk_states)
            chunk_values = policy_model.jepa_model.predict_value(latent)[:, 0]
            all_values.append(chunk_values.cpu())
    all_values = torch.cat(all_values)
    
    all_advantages = torch.zeros(len(states_tensor), dtype=torch.float32)
    for i in range(len(states_tensor)):
        val = all_values[i].item()
        turn = bool(turns_np[i])
        z = targets_tensor[i].item()
        
        R = z if turn == chess.WHITE else -z
        V = val if turn == chess.WHITE else -val
        all_advantages[i] = R - V
        
    mean_adv = all_advantages.mean()
    std_adv = all_advantages.std() + 1e-8
    return (all_advantages - mean_adv) / std_adv

def _ppo_step_batch(policy_model, ref_policy_model, batch, optimizer, args, device):
    batch_states, batch_actions, batch_legal_masks, batch_advantages, batch_targets, batch_old_log_probs = [b.to(device) for b in batch]
    
    logits = policy_model(batch_states)
    latent = policy_model.jepa_model.forward_context(batch_states)
    value = policy_model.jepa_model.predict_value(latent)[:, 0]
    
    masked_logits = logits.masked_fill(~batch_legal_masks, float('-inf'))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = F.softmax(masked_logits, dim=-1)
    
    log_prob_selected = log_probs[torch.arange(len(batch_states)), batch_actions]
    ratio = torch.exp(log_prob_selected - batch_old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
    policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
    
    value_loss = F.mse_loss(value, batch_targets)
    
    clean_log_probs = log_probs.masked_fill(~batch_legal_masks, 0.0)
    entropy = -torch.sum(probs * clean_log_probs, dim=-1)
    mean_entropy = torch.mean(entropy)
    
    with torch.no_grad():
        ref_logits = ref_policy_model(batch_states)
        ref_masked_logits = ref_logits.masked_fill(~batch_legal_masks, float('-inf'))
        ref_log_probs = F.log_softmax(ref_masked_logits, dim=-1)
        
    clean_ref_log_probs = ref_log_probs.masked_fill(~batch_legal_masks, 0.0)
    kl_div = torch.sum(probs * (clean_log_probs - clean_ref_log_probs), dim=-1)
    kl_loss = torch.mean(kl_div)
    
    total_loss = policy_loss + args.kl_coeff * kl_loss + args.val_coeff * value_loss - args.entropy_coeff * mean_entropy
    if hasattr(policy_model.policy_head, 'aux_loss'):
        total_loss = total_loss + policy_model.policy_head.aux_loss
        
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return policy_loss.item(), value_loss.item(), mean_entropy.item()

def _run_ppo_epoch(policy_model, ref_policy_model, rl_loader, optimizer, args, device):
    policy_model.train()
    total_epoch_policy_loss = 0.0
    total_epoch_value_loss = 0.0
    total_epoch_entropy = 0.0
    
    train_pbar = tqdm(rl_loader, desc="Batched Optimization", leave=False, mininterval=30.0)
    for batch in train_pbar:
        p_loss, v_loss, ent = _ppo_step_batch(policy_model, ref_policy_model, batch, optimizer, args, device)
        total_epoch_policy_loss += p_loss
        total_epoch_value_loss += v_loss
        total_epoch_entropy += ent
    train_pbar.close()
    
    n = len(rl_loader)
    return total_epoch_policy_loss / n, total_epoch_value_loss / n, total_epoch_entropy / n

def train_rl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    policy_model, ref_policy_model, jepa_args = _load_rl_models(args, device)
    
    optim_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    
    temp_checkpoint_path = "JEPA_RL/temp_policy_epoch.pth"
    print("\nStarting Self-Play RL loop...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} - Generating {args.games_per_epoch} games...")
        os.makedirs(os.path.dirname(temp_checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': policy_model.state_dict(),
            'jepa_args': jepa_args,
            'head_type': policy_model.head_type
        }, temp_checkpoint_path)
        
        self_play_res = _collect_multiprocess_self_play(args, temp_checkpoint_path, device)
        if self_play_res is None:
            print("  ⚠️ Warning: No moves collected. Skipping training.")
            continue
        states_np, actions_np, turns_np, legal_masks_np, targets_np, old_lp_np, draw_stats = self_play_res
        avg_len, w_wins, b_wins, draws = draw_stats
        print(f"  Summary: Avg Length: {avg_len:.1f} | White Wins: {w_wins} | Black Wins: {b_wins} | Draws: {draws}")
        
        states_tensor = torch.from_numpy(states_np)
        targets_tensor = torch.from_numpy(targets_np)
        advantages = _compute_advantages(policy_model, states_tensor, turns_np, targets_tensor, device)
        
        rl_dataset = ChessRlDataset(
            states_tensor, torch.from_numpy(actions_np), torch.from_numpy(legal_masks_np),
            advantages, targets_tensor, torch.from_numpy(old_lp_np)
        )
        rl_loader = DataLoader(rl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        avg_pol, avg_val, avg_ent = _run_ppo_epoch(policy_model, ref_policy_model, rl_loader, optimizer, args, device)
        print(f"  Losses - Policy: {avg_pol:.4f} | Value: {avg_val:.4f} | Entropy: {avg_ent:.4f}")
        
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': policy_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'jepa_args': jepa_args, 'head_type': policy_model.head_type, 'args': args
        }, args.save_path)
        print(f"RL policy checkpoint saved to '{args.save_path}'")
        
        if os.path.exists(temp_checkpoint_path):
            try: os.remove(temp_checkpoint_path)
            except OSError: pass
    print("\n🎉 Reinforcement learning training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chess Policy via Self-Play RL.")
    parser.add_argument('--jepa_checkpoint', type=str, default='JEPA/chess_jepa.pth', help="Path to pre-trained JEPA model.")
    parser.add_argument('--policy_checkpoint', type=str, default='JEPA_Policy/chess_sft_policy.pth', help="Path to base policy model.")
    parser.add_argument('--games_per_epoch', type=int, default=200, help="Number of self-play games per epoch.")
    parser.add_argument('--epochs', type=int, default=300, help="Number of RL epochs.")
    parser.add_argument('--max_moves', type=int, default=200, help="Max moves per game before draw.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Exploration temperature for self-play.")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay.")
    parser.add_argument('--val_coeff', type=float, default=1.0, help="Loss coefficient for the value head.")
    parser.add_argument('--entropy_coeff', type=float, default=0.001, help="Loss coefficient for policy entropy.")
    parser.add_argument('--clip_eps', type=float, default=0.2, help="PPO clipping epsilon.")
    parser.add_argument('--kl_coeff', type=float, default=0.02, help="KL penalty coefficient against reference policy.")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for RL targets.")
    parser.add_argument('--opp_sft', type=float, default=0.2, help="Probability of playing against SFT model.")
    parser.add_argument('--opp_random', type=float, default=0.05, help="Probability of playing against Random player.")
    parser.add_argument('--opp_stockfish', type=float, default=0.25, help="Probability of playing against Stockfish.")
    parser.add_argument('--opp_self', type=float, default=0.5, help="Probability of playing against Self (RL model).")
    parser.add_argument('--stockfish_path', type=str, default='stockfish_bin/stockfish/stockfish-ubuntu-x86-64', help="Path to Stockfish binary.")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze JEPA encoder backbone.")
    parser.add_argument('--save_path', type=str, default='JEPA_RL/chess_rl_policy.pth', help="Path to save RL checkpoint.")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for RL training.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader.")
    parser.add_argument('--num_selfplay_workers', type=int, default=4, help="Number of parallel multiprocessing self-play worker processes.")
    parser.add_argument('--head_type', type=str, default='linear', choices=['linear', 'mlp', 'transformer', 'moe', 'latent_thinker', 'moe_latent_thinker'],
                        help="Policy head architecture: linear (default), mlp, transformer, moe, latent_thinker, or moe_latent_thinker.")
    
    args = parser.parse_args()
    train_rl(args)
