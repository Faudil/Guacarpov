import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = v = src2
        src2, _ = self.self_attn(q, k, v)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class ViTEncoder(nn.Module):
    """
    Vision Transformer that treats the 8x8 chess board as 64 square tokens.
    """
    def __init__(self, in_channels=111, latent_dim=512, d_model=512, nhead=8, num_layers=8):
        super().__init__()
        # Project each square's features into d_model
        self.square_proj = nn.Linear(in_channels, d_model)
        
        # 2D Positional Embeddings
        # 64 squares + 1 CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, 65, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerCustom(d_model, nhead, dim_feedforward=d_model * 4) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, latent_dim)
        
    def forward(self, x):
        # x: (B, 111, 8, 8)
        B = x.shape[0]
        
        # Flatten spatial dims: (B, 111, 64) -> (B, 64, 111)
        x = x.view(B, x.shape[1], -1).transpose(1, 2)
        
        # Project: (B, 64, d_model)
        x = self.square_proj(x)
        
        # Prepend CLS token: (B, 65, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Apply Transformer layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Return the CLS token representation as the global latent projected to latent_dim
        return self.proj(x[:, 0])

class ViTPredictor(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=1024, action_dim=4672, action_embed_dim=64):
        super().__init__()
        
        self.action_embed = nn.Embedding(action_dim + 1, action_embed_dim, padding_idx=action_dim)
        self.action_proj = nn.Linear(action_embed_dim, latent_dim)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x, actions=None):
        if actions is not None:
            actions_clean = actions.clone()
            actions_clean[actions_clean == -1] = 4672
            a_emb = self.action_embed(actions_clean)
            a_proj = self.action_proj(a_emb)
            x = x + a_proj
            
        return self.net(x)

class ChessJEPA_ViT(nn.Module):
    """
    Option B: Vision Transformer JEPA.
    Provides infinite scalability and global attention across the board.
    """
    def __init__(self, in_channels=111, latent_dim=512, d_model=512, nhead=8, num_layers=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.context_encoder = ViTEncoder(in_channels, latent_dim, d_model, nhead, num_layers)
        self.target_encoder = ViTEncoder(in_channels, latent_dim, d_model, nhead, num_layers)
        
        self._copy_weights(self.context_encoder, self.target_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = ViTPredictor(latent_dim, d_model * 2)
        
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    @staticmethod
    def _copy_weights(source, target):
        target.load_state_dict(source.state_dict())
        
    @torch.no_grad()
    def update_target_encoder(self, decay=0.996):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1.0 - decay)
            
    def forward_context(self, board):
        return self.context_encoder(board)
        
    def forward_target(self, next_board):
        with torch.no_grad():
            return self.target_encoder(next_board)
            
    def forward_predict(self, context_latent, actions=None):
        return self.predictor(context_latent, actions)
        
    def predict_value(self, context_latent):
        return self.value_head(context_latent)
