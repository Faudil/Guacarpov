import torch
import torch.nn as nn

from .latent_thinker_head import ACTCell
from .moe_head import MoEPolicyHead

class MoELatentThinkingHead(nn.Module):
    """
    Combines the ACT loop from HybridLatentThinkingHead with the sparse routing of MoEPolicyHead.
    """
    def __init__(self, latent_dim: int = 256, max_steps: int = 10, jepa_model=None, 
                 ponder_cost_coeff: float = 0.01, num_experts: int = 8, 
                 hidden_dim: int = 512, top_k: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.jepa_model = jepa_model
        self.ponder_cost_coeff = ponder_cost_coeff
        
        self.act_cell = ACTCell(latent_dim=latent_dim)
        # We reuse the MoE Policy Head as our final projector instead of a Linear layer
        self.moe_head = MoEPolicyHead(
            latent_dim=latent_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k,
            load_balance_coeff=0.01
        )
        
        self._aux_loss = torch.tensor(0.0)

    @property
    def aux_loss(self) -> torch.Tensor:
        """Sum of ponder cost and MoE load balancing loss."""
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
        thought_state = latent.unsqueeze(1) # Initial query is the current state
        
        accumulated_thought = torch.zeros_like(latent)
        accumulated_prob = torch.zeros(B, 1, device=latent.device)
        remainders = torch.ones(B, 1, device=latent.device)
        
        for step in range(self.max_steps):
            if self.jepa_model is not None and hasattr(self.jepa_model, 'predict_value'):
                with torch.no_grad():
                    value_pred = self.jepa_model.predict_value(thought_state.squeeze(1)) # (B, 1)
            else:
                value_pred = torch.zeros(B, 1, device=latent.device)
                
            thought_state, halt_prob, update_gate = self.act_cell(thought_state, trajectory, value_pred)
            
            if step == self.max_steps - 1:
                p_n = remainders
            else:
                p_n = torch.min(halt_prob, remainders)
                
            accumulated_thought = (1.0 - update_gate) * accumulated_thought + update_gate * p_n * thought_state.squeeze(1)
            
            accumulated_prob = accumulated_prob + p_n
            remainders = remainders - p_n
            
            if (remainders <= 0).all():
                break
                
        # 3. Route accumulated thought through MoE
        logits = self.moe_head(accumulated_thought)
        
        # 4. Store combined auxiliary loss
        if self.training:
            ponder_loss = self.ponder_cost_coeff * accumulated_prob.mean()
            self._aux_loss = ponder_loss + self.moe_head.aux_loss
        else:
            self._aux_loss = torch.tensor(0.0, device=latent.device)
            
        return logits
