import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class SpatialEncoder(nn.Module):
    """
    Extracts a spatial latent vector for a single board frame.
    Input: (B*T, C, 8, 8) -> Output: (B*T, d_model)
    """
    def __init__(self, in_channels=15, d_model=512, dim=128, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * 8 * 8, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.head(x)

class SpatioTemporalEncoder(nn.Module):
    """
    Processes a sequence of historical states.
    Input: (B, T, C, 8, 8)
    Output: (B, d_model) — representing the context trajectory.
    """
    def __init__(self, in_channels=15, d_model=512, t_history=8, nhead=8, num_layers=4, dim=128, num_blocks=4):
        super().__init__()
        self.spatial = SpatialEncoder(in_channels=in_channels, d_model=d_model, dim=dim, num_blocks=num_blocks)
        
        # Temporal Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, t_history, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, activation='gelu')
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # We'll use a [CLS] style token, or just pool the temporal sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        # Spatial processing
        x = x.view(B * T, C, H, W)
        spatial_latents = self.spatial(x)
        spatial_latents = spatial_latents.view(B, T, -1) # (B, T, d_model)
        
        # Temporal processing
        spatial_latents = spatial_latents + self.pos_embedding[:, :T, :]
        
        # Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat((cls_tokens, spatial_latents), dim=1) # (B, T+1, d_model)
        
        out = self.temporal(seq) # (B, T+1, d_model)
        
        # Return the aggregated trajectory context
        return self.norm(out[:, 0])

class TrajectoryPredictor(nn.Module):
    """
    Takes the trajectory context latent and predicts the sequence of FUTURE latents.
    Input: Context (B, d_model)
    Output: Future latents (B, T_future, d_model)
    """
    def __init__(self, d_model=512, t_future=4, nhead=8, num_layers=4, action_dim=4672, action_embed_dim=64):
        super().__init__()
        self.t_future = t_future
        self.d_model = d_model
        
        self.action_embed = nn.Embedding(action_dim + 1, action_embed_dim, padding_idx=action_dim)
        self.action_proj = nn.Linear(action_embed_dim, d_model)
        
        # Learned query embeddings for future steps
        self.query_embeds = nn.Parameter(torch.randn(1, t_future, d_model))
        
        # Cross-attention predictor
        # Queries: Future step embeddings
        # Keys/Values: Context latent (projected)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, context_latent, actions=None):
        B = context_latent.shape[0]
        # Context acts as memory (B, 1, d_model)
        memory = context_latent.unsqueeze(1) 
        
        # Queries for the future steps (B, T_future, d_model)
        queries = self.query_embeds.expand(B, -1, -1)
        
        if actions is not None:
            actions_clean = actions.clone()
            actions_clean[actions_clean == -1] = 4672
            a_emb = self.action_embed(actions_clean)
            a_proj = self.action_proj(a_emb)
            queries = queries + a_proj.unsqueeze(1)
            
        out = self.decoder(queries, memory)
        return self.norm(self.out_proj(out))

class ChessJEPA_SpatioTemporal(nn.Module):
    """
    Option C: Spatio-Temporal JEPA (V-JEPA style).
    Models the game as a temporal sequence. Predicts multiple future steps.
    """
    def __init__(self, in_channels=15, d_model=512, t_history=8, t_future=4, dim=128, num_blocks=4, nhead=8, num_layers=4):
        super().__init__()
        self.latent_dim = d_model
        self.t_future = t_future
        
        # Context processes T past frames
        self.context_encoder = SpatioTemporalEncoder(
            in_channels, d_model, t_history, nhead=nhead, num_layers=num_layers, dim=dim, num_blocks=num_blocks
        )
        
        # Target only needs spatial encoder to get ground truth future latents
        self.target_encoder = SpatialEncoder(in_channels, d_model, dim=dim, num_blocks=num_blocks)
        
        self._copy_weights(self.context_encoder.spatial, self.target_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = TrajectoryPredictor(d_model, t_future, nhead=nhead, num_layers=num_layers)
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    @staticmethod
    def _copy_weights(source, target):
        target.load_state_dict(source.state_dict())
        
    @torch.no_grad()
    def update_target_encoder(self, decay=0.996):
        # Update target spatial encoder with EMA of context spatial encoder
        for param_q, param_k in zip(self.context_encoder.spatial.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1.0 - decay)
            
    def forward_context(self, history_boards):
        """ history_boards: (B, T_history, C, H, W) """
        if history_boards.dim() == 4:
            history_boards = history_boards.unsqueeze(1)
        return self.context_encoder(history_boards)
        
    def forward_target(self, future_boards):
        """ future_boards: (B, T_future, C, H, W) """
        if future_boards.dim() == 4:
            future_boards = future_boards.unsqueeze(1)
        B, T, C, H, W = future_boards.shape
        future_boards = future_boards.reshape(B * T, C, H, W)
        with torch.no_grad():
            latents = self.target_encoder(future_boards)
        return latents.view(B, T, self.latent_dim)
        
    def forward_predict(self, context_latent, actions=None):
        """ Predict T_future steps from context latent """
        return self.predictor(context_latent, actions)
        
    def predict_value(self, context_latent):
        return self.value_head(context_latent.detach())
