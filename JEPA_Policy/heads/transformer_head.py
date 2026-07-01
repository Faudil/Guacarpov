import torch
import torch.nn as nn

ACTION_DIM = 4672  # Leela Chess Zero action space size

class TransformerPolicyHead(nn.Module):
    """
    A transformer-based policy head that decomposes the flat latent vector
    into a sequence of spatial tokens and applies multi-head self-attention.
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
