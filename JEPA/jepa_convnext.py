import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block for 2D spatial data.
    Uses depthwise convolutions, LayerNorm instead of BatchNorm, and GELU.
    """
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Pointwise/Inverted bottleneck
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return input + x

class ConvNeXtEncoder(nn.Module):
    """
    Processes the 111-channel game state using a series of ConvNeXt blocks.
    """
    def __init__(self, in_channels=111, latent_dim=512, num_blocks=16, dim=256):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(num_blocks)])
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * 8 * 8, 1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim)
        )
        
    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
            
        # Global average pooling can be used, but for chess preserving the 8x8 spatial grid before flattening is better
        x = x.permute(0, 2, 3, 1) # (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (N, C, H, W)
        
        return self.head(x)

class Predictor(nn.Module):
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

class ChessJEPA_ConvNeXt(nn.Module):
    """
    Option A: Modernized Deep ConvNeXt JEPA.
    Optimized for high-speed RL while having strong spatial inductive biases.
    """
    def __init__(self, in_channels=111, latent_dim=512, num_blocks=16, dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.context_encoder = ConvNeXtEncoder(in_channels, latent_dim, num_blocks, dim)
        self.target_encoder = ConvNeXtEncoder(in_channels, latent_dim, num_blocks, dim)
        
        self._copy_weights(self.context_encoder, self.target_encoder)
        for param in self.target_encoder.parameters():  
            param.requires_grad = False
            
        self.predictor = Predictor(latent_dim)
        
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
