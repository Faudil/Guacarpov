import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_DIM = 4672  # Leela Chess Zero action space size

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
