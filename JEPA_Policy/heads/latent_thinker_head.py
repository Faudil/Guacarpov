import torch
import torch.nn as nn

ACTION_DIM = 4672  # Leela Chess Zero action space size

class ACTCell(nn.Module):
    """
    A single step of Adaptive Computation Time (ACT) reasoning.
    Uses Cross-Attention to query the future trajectory sequence.
    """
    def __init__(self, latent_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        # Value-Guided Halting: Input is thought (latent_dim) + Value (1)
        self.halt_proj = nn.Linear(latent_dim + 1, 1)
        # Gated Correction: Outputs a feature-wise gate (B, latent_dim)
        self.gate_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, thought_state, trajectory_seq, value_pred):
        # thought_state: (B, 1, latent_dim)
        # trajectory_seq: (B, T, latent_dim)
        # value_pred: (B, 1)
        attn_out, _ = self.attn(thought_state, trajectory_seq, trajectory_seq, need_weights=False)
        x = self.norm1(thought_state + attn_out)
        ffn_out = self.ffn(x)
        next_thought = self.norm2(x + ffn_out)
        
        # Concatenate thought and predicted Value
        halt_features = torch.cat([next_thought.squeeze(1), value_pred], dim=-1)
        halt_prob = torch.sigmoid(self.halt_proj(halt_features)) # (B, 1)
        
        # Gated inner monologue
        update_gate = torch.sigmoid(self.gate_proj(next_thought.squeeze(1))) # (B, latent_dim)
        return next_thought, halt_prob, update_gate

class HybridLatentThinkingHead(nn.Module):
    """
    A System-2 Latent Thinking Head.
    
    1. Queries the JEPA model to predict the future trajectory of latents.
    2. Uses an Adaptive Computation Time (ACT) loop to ponder over the sequence
       until it is confident in its move (cumulative halt prob >= 0.99).
    """
    def __init__(self, latent_dim: int = 256, max_steps: int = 10, jepa_model=None, ponder_cost_coeff: float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.jepa_model = jepa_model
        self.ponder_cost_coeff = ponder_cost_coeff
        
        self.act_cell = ACTCell(latent_dim=latent_dim)
        self.output_proj = nn.Linear(latent_dim, ACTION_DIM)
        self._aux_loss = torch.tensor(0.0)

    @property
    def aux_loss(self) -> torch.Tensor:
        """Ponder cost auxiliary loss from the most recent forward pass."""
        return self._aux_loss
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        
        # 1. Rollout future if JEPA is provided and supports it
        if self.jepa_model is not None and hasattr(self.jepa_model, 'forward_predict'):
            with torch.no_grad():
                future_latents = self.jepa_model.forward_predict(latent)
            if future_latents.dim() == 2:
                future_latents = future_latents.unsqueeze(1)
            trajectory = torch.cat([latent.unsqueeze(1), future_latents], dim=1)
        else:
            # Fallback if no JEPA predictor is attached
            trajectory = latent.unsqueeze(1)
            
        # 2. Adaptive Pondering (ACT Loop)
        thought_state = latent.unsqueeze(1)
        
        accumulated_thought = torch.zeros_like(latent)
        accumulated_prob = torch.zeros(B, 1, device=latent.device)
        remainders = torch.ones(B, 1, device=latent.device)
        
        for step in range(self.max_steps):
            # Predict Value for the current thought_state
            if self.jepa_model is not None and hasattr(self.jepa_model, 'predict_value'):
                with torch.no_grad():
                    value_pred = self.jepa_model.predict_value(thought_state.squeeze(1)) # (B, 1)
            else:
                value_pred = torch.zeros(B, 1, device=latent.device)
                
            thought_state, halt_prob, update_gate = self.act_cell(thought_state, trajectory, value_pred)
            
            # If last step, force remaining probability
            if step == self.max_steps - 1:
                p_n = remainders
            else:
                p_n = torch.min(halt_prob, remainders)
                
            # Gated Correction
            accumulated_thought = (1.0 - update_gate) * accumulated_thought + update_gate * p_n * thought_state.squeeze(1)
            
            accumulated_prob = accumulated_prob + p_n
            remainders = remainders - p_n
            
            # Break early if all items in batch have halted
            if (remainders <= 0).all():
                break
                
        # Store ponder cost
        if self.training:
            self._aux_loss = self.ponder_cost_coeff * accumulated_prob.mean()
        else:
            self._aux_loss = torch.tensor(0.0, device=latent.device)

        # 3. Projection to Action Logits
        return self.output_proj(accumulated_thought)
