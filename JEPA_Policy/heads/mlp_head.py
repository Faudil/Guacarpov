import torch
import torch.nn as nn

ACTION_DIM = 4672  # Leela Chess Zero action space size

class MLPPolicyHead(nn.Module):
    """
    A deep multi-layer perceptron policy head with residual skip connections.
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
