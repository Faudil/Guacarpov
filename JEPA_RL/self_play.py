import torch
import torch.nn.functional as F
import chess
import numpy as np
import os
import sys
from tqdm import tqdm

# Add JEPA and JEPA_Policy folders to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "JEPA_Policy"))

from policy_model import ChessJepaPolicy, board_to_tensor

def _init_self_play_games(num_games, opp_probs):
    if opp_probs is None:
        opp_probs = {'sft': 1.0, 'random': 0.0, 'stockfish': 0.0, 'self': 0.0}
    boards = [chess.Board() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
    trajectories = [[] for _ in range(num_games)]
    outcomes = [0.0] * num_games
    active_indices = list(range(num_games))
    
    opp_types = np.random.choice(
        ['sft', 'random', 'stockfish', 'self'], 
        size=num_games, 
        p=[opp_probs['sft'], opp_probs['random'], opp_probs['stockfish'], opp_probs['self']]
    )
    rl_colors = [chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(num_games)]
    return boards, histories, trajectories, outcomes, active_indices, opp_types, rl_colors

def _get_batch_predictions(policy_model, sft_policy_model, active_boards, active_indices, histories, rl_indices, sft_indices, device):
    fmt = '111' if policy_model._get_expected_channels() > 12 else '12'
    board_tensors = torch.stack([
        board_to_tensor(active_boards[idx], history=histories[active_indices[idx]], format=fmt) 
        for idx in range(len(active_indices))
    ]).to(device)
    
    logits_batch = torch.full((len(active_indices), 4672), float('-inf'), device=device)
    current_values = np.zeros(len(active_indices), dtype=np.float32)
    
    with torch.no_grad():
        if rl_indices:
            rl_b = board_tensors[rl_indices]
            logits_batch[rl_indices] = policy_model(rl_b)
            rl_latent = policy_model.jepa_model.forward_context(rl_b)
            current_values[rl_indices] = policy_model.jepa_model.predict_value(rl_latent)[:, 0].cpu().numpy()
            
        if sft_indices and sft_policy_model is not None:
            sft_b = board_tensors[sft_indices]
            logits_batch[sft_indices] = sft_policy_model(sft_b)
            sft_latent = sft_policy_model.jepa_model.forward_context(sft_b)
            current_values[sft_indices] = sft_policy_model.jepa_model.predict_value(sft_latent)[:, 0].cpu().numpy()
            
    return board_tensors, logits_batch, current_values

def _evaluate_legal_moves(board, mapper):
    legal_moves = list(board.legal_moves)
    legal_indices = []
    valid_moves = []
    for m in legal_moves:
        try:
            legal_indices.append(mapper.move_to_index(m))
            valid_moves.append(m)
        except ValueError:
            continue
    return valid_moves, legal_indices

def _query_stockfish(board, valid_moves, stockfish_engine, mapper):
    try:
        import chess.engine as ce
        result = stockfish_engine.play(board, ce.Limit(time=0.01, depth=2))
        return result.move
    except Exception:
        return np.random.choice(valid_moves)

def _apply_lookahead(board, valid_moves, legal_indices, val, masked_logits):
    active_value = val if board.turn == chess.WHITE else -val
    for move, idx in zip(valid_moves, legal_indices):
        next_board = board.copy()
        next_board.push(move)
        if next_board.is_game_over(claim_draw=True):
            if next_board.is_checkmate():
                masked_logits[idx] = 100.0
            else:
                masked_logits[idx] -= (active_value * 15.0)

def _filter_removed_games(active_indices, games_to_remove, masked_logits_batch, has_logit_sample, sampled_indices, current_values, legal_masks, valid_moves_per_game):
    keep_indices = [i for i, g in enumerate(active_indices) if g not in games_to_remove]
    active_indices = [g for g in active_indices if g not in games_to_remove]
    if not active_indices:
        return active_indices, None, None, None, None, None, None
    return (
        active_indices,
        masked_logits_batch[keep_indices],
        has_logit_sample[keep_indices],
        sampled_indices[keep_indices],
        current_values[keep_indices],
        [legal_masks[i] for i in keep_indices],
        [valid_moves_per_game[i] for i in keep_indices]
    )

def _sample_batch_actions(masked_logits_batch, probs_batch, has_logit_sample, sampled_indices, temp):
    if has_logit_sample.any():
        nn_probs = probs_batch[has_logit_sample]
        try:
            nn_samples = torch.multinomial(nn_probs, 1)[:, 0]
        except RuntimeError:
            nn_samples = torch.argmax(masked_logits_batch[has_logit_sample], dim=-1)
        sampled_indices[has_logit_sample] = nn_samples

def _step_active_games(active_indices, boards, sampled_indices, valid_moves_per_game, rl_colors, opp_types, mapper, board_tensors, legal_masks, old_log_probs, trajectories, outcomes):
    next_active_indices = []
    for idx_in_batch, game_idx in enumerate(active_indices):
        board = boards[game_idx]
        sampled_idx = int(sampled_indices[idx_in_batch])
        valid_moves, legal_indices = valid_moves_per_game[idx_in_batch]
        is_rl_turn = (board.turn == rl_colors[game_idx])
        
        try:
            chosen_move = mapper.index_to_move(sampled_idx, board)
        except Exception:
            chosen_move = np.random.choice(valid_moves)
            sampled_idx = mapper.move_to_index(chosen_move)
            
        board_t_cpu = board_tensors[idx_in_batch].cpu().to(torch.uint8)
        legal_mask = legal_masks[idx_in_batch]
        
        if is_rl_turn or opp_types[game_idx] == 'self':
            trajectories[game_idx].append((
                board_t_cpu,
                sampled_idx,
                board.turn,
                legal_mask,
                old_log_probs[idx_in_batch]
            ))
        board.push(chosen_move)
        
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                outcomes[game_idx] = 1.0 if outcome.winner == chess.WHITE else -1.0
            else:
                outcomes[game_idx] = 0.0
        else:
            next_active_indices.append(game_idx)
    return next_active_indices

def _self_play_step(active_indices, boards, histories, trajectories, outcomes, opp_types, rl_colors, policy_model, sft_policy_model, temp, device, stockfish_engine, mapper):
    active_boards = [boards[i] for i in active_indices]
    
    rl_indices = [idx for idx, g in enumerate(active_indices) if (boards[g].turn == rl_colors[g]) or opp_types[g] == 'self']
    sft_indices = [idx for idx, g in enumerate(active_indices) if (boards[g].turn != rl_colors[g]) and opp_types[g] == 'sft']
    random_indices = [idx for idx, g in enumerate(active_indices) if (boards[g].turn != rl_colors[g]) and opp_types[g] == 'random']
    stockfish_indices = [idx for idx, g in enumerate(active_indices) if (boards[g].turn != rl_colors[g]) and opp_types[g] == 'stockfish']
    
    board_tensors, logits_batch, current_values = _get_batch_predictions(
        policy_model, sft_policy_model, active_boards, active_indices, histories, rl_indices, sft_indices, device
    )
    
    masked_logits_batch = logits_batch.clone()
    legal_masks, valid_moves_per_game, games_to_remove = [], [], []
    sampled_indices = torch.zeros(len(active_indices), dtype=torch.long, device=device)
    has_logit_sample = torch.zeros(len(active_indices), dtype=torch.bool, device=device)
    
    for idx_in_batch, game_idx in enumerate(active_indices):
        board = boards[game_idx]
        valid_moves, legal_indices = _evaluate_legal_moves(board, mapper)
        if not legal_indices:
            games_to_remove.append(game_idx)
            continue
            
        legal_mask = torch.zeros(4672, dtype=torch.bool)
        legal_mask[legal_indices] = True
        legal_masks.append(legal_mask)
        valid_moves_per_game.append((valid_moves, legal_indices))
        
        if idx_in_batch in random_indices:
            chosen_move = np.random.choice(valid_moves)
            sampled_indices[idx_in_batch] = mapper.move_to_index(chosen_move)
        elif idx_in_batch in stockfish_indices and stockfish_engine is not None:
            chosen_move = _query_stockfish(board, valid_moves, stockfish_engine, mapper)
            sampled_indices[idx_in_batch] = mapper.move_to_index(chosen_move)
        else:
            has_logit_sample[idx_in_batch] = True
            masked_logits_batch[idx_in_batch][~legal_mask] = float('-inf')
            _apply_lookahead(board, valid_moves, legal_indices, current_values[idx_in_batch], masked_logits_batch[idx_in_batch])
            
    if games_to_remove:
        active_indices, masked_logits_batch, has_logit_sample, sampled_indices, current_values, legal_masks, valid_moves_per_game = _filter_removed_games(
            active_indices, games_to_remove, masked_logits_batch, has_logit_sample, sampled_indices, current_values, legal_masks, valid_moves_per_game
        )
        if not active_indices:
            return active_indices
            
    probs_batch = F.softmax(masked_logits_batch / temp, dim=-1)
    log_probs_batch = F.log_softmax(masked_logits_batch / temp, dim=-1)
    
    _sample_batch_actions(masked_logits_batch, probs_batch, has_logit_sample, sampled_indices, temp)
    
    old_log_probs = log_probs_batch[torch.arange(len(sampled_indices)), sampled_indices].cpu().numpy()
    sampled_indices = sampled_indices.cpu().numpy()
    
    return _step_active_games(
        active_indices, boards, sampled_indices, valid_moves_per_game, rl_colors, opp_types, mapper, 
        board_tensors, legal_masks, old_log_probs, trajectories, outcomes
    )

def self_play_games_batched(policy_model, sft_policy_model, num_games, temp=1.0, max_moves=200, device='cpu', opp_probs=None, stockfish_engine=None):
    boards, histories, trajectories, outcomes, active_indices, opp_types, rl_colors = _init_self_play_games(num_games, opp_probs)
    
    policy_model.eval()
    if sft_policy_model is not None:
        sft_policy_model.eval()
    move_count = 0
    mapper = policy_model.move_mapper
    
    pbar = tqdm(total=max_moves, desc="Self-play plies", leave=False, mininterval=30.0)
    
    while active_indices and move_count < max_moves:
        active_indices = _self_play_step(
            active_indices, boards, histories, trajectories, outcomes, opp_types, rl_colors,
            policy_model, sft_policy_model, temp, device, stockfish_engine, mapper
        )
        move_count += 1
        pbar.update(1)
        
    pbar.close()
    return trajectories, outcomes

def _load_worker_policies(checkpoint_path, sft_checkpoint_path, device):
    from factory import build_jepa_model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    jepa_args = checkpoint['jepa_args']
    head_type = checkpoint.get('head_type', 'linear')
    
    jepa_model = build_jepa_model(jepa_args).to(device)
    policy_model = ChessJepaPolicy(jepa_model, freeze_jepa=True, head_type=head_type).to(device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    
    sft_policy_model = None
    if sft_checkpoint_path and os.path.exists(sft_checkpoint_path):
        sft_ckpt = torch.load(sft_checkpoint_path, map_location=device, weights_only=False)
        sft_jepa_args = sft_ckpt.get('jepa_args', sft_ckpt.get('args', jepa_args))
        sft_head_type = sft_ckpt.get('head_type', head_type)
        sft_jepa = build_jepa_model(sft_jepa_args).to(device)
        sft_policy_model = ChessJepaPolicy(sft_jepa, freeze_jepa=True, head_type=sft_head_type).to(device)
        sft_policy_model.load_state_dict(sft_ckpt['model_state_dict'])
        sft_policy_model.eval()
        
    return policy_model, sft_policy_model, jepa_args

def _flatten_worker_trajectories(trajectories, outcomes, gamma):
    all_states, all_actions, all_turns, all_legal_masks, all_targets, all_old_log_probs = [], [], [], [], [], []
    for traj, z in zip(trajectories, outcomes):
        T = len(traj)
        for t, (board_t_cpu, sampled_idx, turn, legal_mask, old_lp) in enumerate(traj):
            discounted_z = z * (gamma ** (T - 1 - t))
            all_states.append(board_t_cpu.numpy())
            all_actions.append(sampled_idx)
            all_turns.append(turn)
            all_legal_masks.append(legal_mask.numpy())
            all_targets.append(discounted_z)
            all_old_log_probs.append(old_lp)
            
    if not all_states:
        return None
        
    return (
        np.array(all_states, dtype=np.uint8),
        np.array(all_actions, dtype=np.int64),
        np.array(all_turns, dtype=np.bool_),
        np.array(all_legal_masks, dtype=np.bool_),
        np.array(all_targets, dtype=np.float32),
        np.array(all_old_log_probs, dtype=np.float32),
        outcomes
    )

def self_play_worker(worker_id, num_games, temp, max_moves, checkpoint_path, device_str, gamma=0.99, sft_checkpoint_path=None, stockfish_path=None, opp_probs=None):
    device = torch.device(device_str)
    policy_model, sft_policy_model, _ = _load_worker_policies(checkpoint_path, sft_checkpoint_path, device)
    
    stockfish_engine = None
    if stockfish_path and os.path.exists(stockfish_path) and opp_probs and opp_probs.get('stockfish', 0) > 0:
        import chess.engine as ce
        stockfish_engine = ce.SimpleEngine.popen_uci(stockfish_path)
        
    trajectories, outcomes = self_play_games_batched(
        policy_model=policy_model,
        sft_policy_model=sft_policy_model,
        num_games=num_games,
        temp=temp,
        max_moves=max_moves,
        device=device,
        opp_probs=opp_probs,
        stockfish_engine=stockfish_engine
    )
    
    if stockfish_engine:
        stockfish_engine.quit()
        
    return _flatten_worker_trajectories(trajectories, outcomes, gamma)
