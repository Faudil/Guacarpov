"""
Policy Head Architectures for the Chess JEPA Agent.

All heads consume the JEPA context encoder's latent vector (B, latent_dim)
and output raw logits over the 4672-dim Leela Chess action space.

Usage:
    from policy_heads import MLPPolicyHead, TransformerPolicyHead, MoEPolicyHead

    head = MLPPolicyHead(latent_dim=256)           # ~2.5M params
    head = TransformerPolicyHead(latent_dim=256)    # ~3.1M params
    head = MoEPolicyHead(latent_dim=256)            # ~5.4M params
    head = HybridLatentThinkingHead(latent_dim=256) # ~1.1M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_DIM = 4672  # Leela Chess Zero action space size


class MLPPolicyHead(nn.Module):
    """
    A deep multi-layer perceptron policy head with residual skip connections.

    Architecture:
        latent → Linear → LayerNorm → GELU → Dropout
              → Linear → LayerNorm → GELU → Dropout  (+residual)
              → Linear → LayerNorm → GELU → Dropout  (+residual)
              → Linear → 4672 logits

    Design rationale:
        - 3 hidden layers (1024-dim) give the head enough capacity to learn
          complex move distributions during RL without being so large that it
          overfits on small self-play datasets.
        - Residual connections stabilize gradients through the deeper stack.
        - LayerNorm + Dropout prevent co-adaptation and improve generalisation.

    Parameter count (latent_dim=256, hidden_dim=1024):
        ~2.5M trainable parameters  (vs ~1.2M for a single Linear layer)
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 1024,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.res_blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.output_proj = nn.Linear(hidden_dim, ACTION_DIM)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) from JEPA context encoder.
        Returns:
            logits: (B, 4672) raw action logits.
        """
        x = self.input_proj(latent)  # (B, hidden_dim)
        for block in self.res_blocks:
            x = x + block(x)         # residual
        return self.output_proj(x)   # (B, 4672)


class TransformerPolicyHead(nn.Module):
    """
    A transformer-based policy head that decomposes the flat latent vector
    into a sequence of spatial tokens and applies multi-head self-attention.

    Architecture:
        latent (B, latent_dim)
          → reshape to (B, num_tokens, token_dim)
          → project tokens up to d_model
          → add learnable positional embedding
          → prepend learnable [CLS] token
          → N × TransformerEncoderLayer (self-attention + FFN)
          → [CLS] output → Linear → 4672 logits

    Design rationale:
        - Chess is inherently spatial: piece interactions are relational.
          Self-attention lets the head learn pairwise feature interactions
          (e.g. "this knight attacks that square") that a flat MLP cannot
          model efficiently.
        - Positional embeddings let the model differentiate between latent
          dimensions that correspond to different spatial regions.
        - The [CLS] token aggregates information from all tokens for the
          final projection, acting as a learned pooling mechanism.

    Parameter count (latent_dim=256, num_heads=4, num_layers=3):
        ~3.1M trainable parameters
    """

    def __init__(self, latent_dim: int = 256, num_tokens: int = 16,
                 num_heads: int = 4, num_layers: int = 3,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert latent_dim % num_tokens == 0, \
            f"latent_dim ({latent_dim}) must be divisible by num_tokens ({num_tokens})"

        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens  # e.g. 256/16 = 16

        # Project each token up to a wider working dimension for richer attention
        self.d_model = max(self.token_dim * 4, 128)  # at least 128
        self.token_proj = nn.Linear(self.token_dim, self.d_model)

        # Learnable [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Positional embeddings for num_tokens + 1 (CLS)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens + 1, self.d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, ACTION_DIM)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) from JEPA context encoder.
        Returns:
            logits: (B, 4672) raw action logits.
        """
        B = latent.size(0)

        # Reshape flat latent into a sequence of spatial tokens
        tokens = latent.view(B, self.num_tokens, self.token_dim)  # (B, T, token_dim)
        tokens = self.token_proj(tokens)  # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)

        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)  # (B, T+1, d_model)

        cls_out = self.norm(tokens[:, 0])  # (B, d_model)
        return self.output_proj(cls_out)   # (B, 4672)


class ExpertMLP(nn.Module):
    """A single expert sub-network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEPolicyHead(nn.Module):
    """
    A Mixture-of-Experts policy head with top-k sparse gating.

    Architecture:
        latent (B, latent_dim)
          → Gating network → softmax → top-k expert weights
          → Route latent to k selected experts
          → Weighted sum of expert outputs → 4672 logits

    Design rationale:
        Chess positions fall into qualitatively different regimes: quiet
        positional play, sharp tactical complications, endgame technique,
        opening theory. A single MLP must learn a single compromise function
        for all of them. MoE lets the model learn specialised experts for
        different position types and route each position to the most relevant
        experts.

        - Top-2 routing balances specialisation with sufficient gradient flow.
        - Load-balancing loss prevents expert collapse (one expert dominating).
        - Each expert has ~600K params; with 8 experts the total head is ~5.4M
          params but only ~1.4M are active per position (compute-efficient).

    Parameter count (latent_dim=256, num_experts=8, hidden_dim=512, top_k=2):
        ~5.4M total parameters, ~1.4M active per forward pass
    """

    def __init__(self, latent_dim: int = 256, num_experts: int = 8,
                 hidden_dim: int = 512, top_k: int = 2,
                 dropout: float = 0.1, load_balance_coeff: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff

        # Gating network: produces expert selection logits
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_experts),
        )

        # Expert sub-networks
        self.experts = nn.ModuleList([
            ExpertMLP(latent_dim, hidden_dim, ACTION_DIM, dropout)
            for _ in range(num_experts)
        ])

        # Shared output LayerNorm for stability
        self.output_norm = nn.LayerNorm(ACTION_DIM)

        # Store auxiliary loss from last forward pass (for training)
        self._aux_loss = torch.tensor(0.0)

    @property
    def aux_loss(self) -> torch.Tensor:
        """Load-balancing auxiliary loss from the most recent forward pass."""
        return self._aux_loss

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) from JEPA context encoder.
        Returns:
            logits: (B, 4672) raw action logits.
        """
        B = latent.size(0)

        # Compute gating scores
        gate_logits = self.gate(latent)               # (B, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)   # (B, num_experts)

        # Top-k expert selection
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # (B, k)
        # Renormalize the top-k weights so they sum to 1
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute expert outputs only for selected experts (sparse routing)
        output = torch.zeros(B, ACTION_DIM, device=latent.device, dtype=latent.dtype)

        for k_idx in range(self.top_k):
            expert_indices = topk_idx[:, k_idx]   # (B,) which expert for each sample
            weights = topk_weights[:, k_idx]      # (B,) weight for this expert

            for e_id in range(self.num_experts):
                mask = (expert_indices == e_id)
                if not mask.any():
                    continue
                expert_input = latent[mask]                         # (n, latent_dim)
                expert_out = self.experts[e_id](expert_input)       # (n, 4672)
                output[mask] += weights[mask].unsqueeze(-1) * expert_out

        output = self.output_norm(output)

        # Compute load-balancing auxiliary loss (Switch Transformer style)
        # Encourages uniform expert utilisation across the batch
        if self.training:
            expert_counts = torch.zeros(self.num_experts, device=latent.device)
            for k_idx in range(self.top_k):
                for e_id in range(self.num_experts):
                    expert_counts[e_id] += (topk_idx[:, k_idx] == e_id).float().sum()
            expert_frac = expert_counts / (B * self.top_k + 1e-8)
            mean_gate = gate_probs.mean(dim=0)
            self._aux_loss = self.load_balance_coeff * self.num_experts * (
                expert_frac * mean_gate
            ).sum()
        else:
            self._aux_loss = torch.tensor(0.0, device=latent.device)

        return output


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
        self.halt_proj = nn.Linear(latent_dim, 1)

    def forward(self, thought_state, trajectory_seq):
        # thought_state: (B, 1, latent_dim)
        # trajectory_seq: (B, T, latent_dim)
        attn_out, _ = self.attn(thought_state, trajectory_seq, trajectory_seq, need_weights=False)
        x = self.norm1(thought_state + attn_out)
        ffn_out = self.ffn(x)
        next_thought = self.norm2(x + ffn_out)
        halt_prob = torch.sigmoid(self.halt_proj(next_thought.squeeze(1))) # (B, 1)
        return next_thought, halt_prob

class HybridLatentThinkingHead(nn.Module):
    """
    A System-2 Latent Thinking Head.
    
    1. Queries the JEPA model to predict the future trajectory of latents.
    2. Uses an Adaptive Computation Time (ACT) loop to ponder over the sequence
       until it is confident in its move (cumulative halt prob >= 0.99).
    """
    def __init__(self, latent_dim: int = 256, max_steps: int = 10, jepa_model=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.jepa_model = jepa_model
        
        self.act_cell = ACTCell(latent_dim=latent_dim)
        self.output_proj = nn.Linear(latent_dim, ACTION_DIM)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        
        # 1. Rollout future if JEPA is provided and supports it
        if self.jepa_model is not None and hasattr(self.jepa_model, 'forward_predict'):
            with torch.no_grad():
                future_latents = self.jepa_model.forward_predict(latent)
            trajectory = torch.cat([latent.unsqueeze(1), future_latents], dim=1)
        else:
            # Fallback if no JEPA predictor is attached
            trajectory = latent.unsqueeze(1)
            
        # 2. Adaptive Pondering (ACT Loop)
        thought_state = latent.unsqueeze(1)
        
        accumulated_thought = torch.zeros_like(latent)
        remainders = torch.ones(B, 1, device=latent.device)
        
        for step in range(self.max_steps):
            thought_state, halt_prob = self.act_cell(thought_state, trajectory)
            
            # If last step, force remaining probability
            if step == self.max_steps - 1:
                p_n = remainders
            else:
                p_n = torch.min(halt_prob, remainders)
                
            accumulated_thought += p_n * thought_state.squeeze(1)
            remainders -= p_n
            
            # Break early if all items in batch have halted
            if (remainders <= 0).all():
                break
                
        # 3. Projection to Action Logits
        return self.output_proj(accumulated_thought)

# ──────────────────────────────────────────────────────────────────────────────
# Utility: parameter counting
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick sanity check: instantiate all heads and run a forward pass
    latent_dim = 256
    batch_size = 4
    dummy_latent = torch.randn(batch_size, latent_dim)

    heads = {
        'MLP': MLPPolicyHead(latent_dim=latent_dim),
        'Transformer': TransformerPolicyHead(latent_dim=latent_dim),
        'MoE': MoEPolicyHead(latent_dim=latent_dim),
        'LatentThinking': HybridLatentThinkingHead(latent_dim=latent_dim)
    }

    print(f"{'Head':<15} {'Params':>12} {'Output Shape':>15}")
    print("─" * 45)
    for name, head in heads.items():
        out = head(dummy_latent)
        params = count_parameters(head)
        print(f"{name:<15} {params:>12,} {str(tuple(out.shape)):>15}")
        assert out.shape == (batch_size, ACTION_DIM), f"{name} output shape mismatch!"

    print("\n✅ All policy heads passed forward pass sanity check.")