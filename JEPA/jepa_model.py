import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ResidualBlock(nn.Module):
    """
    A standard residual block with 2D convolutions, batch normalization,
    and a skip connection.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessEncoder(nn.Module):
    """
    Encodes a (12, 8, 8) chess board representation into a latent vector.
    Used for both the Context Encoder and Target Encoder.
    """
    def __init__(self, in_channels=12, latent_dim=256, num_res_blocks=4, num_filters=64):
        super(ChessEncoder, self).__init__()
        self.conv_input = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Flatten and project to latent space
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 8 * 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, x):
        # Input shape: (B, 12, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    """
    Predicts the next state's latent embedding from the current state's
    latent embedding ONLY — no action conditioning.
    
    This forces the encoder to learn rich strategic representations,
    since the predictor must infer likely continuations from board features alone.
    """
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=4672, action_embed_dim=64):
        super(Predictor, self).__init__()
        
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
        
    def forward(self, context_latent, actions=None):
        if actions is not None:
            actions_clean = actions.clone()
            actions_clean[actions_clean == -1] = 4672
            a_emb = self.action_embed(actions_clean)
            a_proj = self.action_proj(a_emb)
            context_latent = context_latent + a_proj
            
        return self.net(context_latent)

class ChessJEPA(nn.Module):
    """
    Chess JEPA v2 — True Joint Embedding Predictive Architecture.
    
    Learns to predict the latent representation of the next board state
    from the latent representation of the current board state, without
    knowing what move was played.
    
    Collapse prevention: BYOL-style EMA on the predictor weights.
    The target encoder uses EMA of the context encoder (standard JEPA).
    The predictor also uses an EMA copy for stable targets.
    """
    def __init__(self, in_channels=12, latent_dim=256, num_res_blocks=4, num_filters=64):
        super(ChessJEPA, self).__init__()
        self.latent_dim = latent_dim
        
        # Context and Target encoders
        self.context_encoder = ChessEncoder(in_channels, latent_dim, num_res_blocks, num_filters)
        self.target_encoder = ChessEncoder(in_channels, latent_dim, num_res_blocks, num_filters)
        self._copy_weights(self.context_encoder, self.target_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Predictor (online, trainable)
        self.predictor = Predictor(latent_dim)
        
        # Value head to estimate game outcome from context latent representation
        # Trained on game outcomes: +1 (white win), 0 (draw), -1 (black win)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    @staticmethod
    def _copy_weights(source, target):
        target.load_state_dict(source.state_dict())
        
    @torch.no_grad()
    def update_target_encoder(self, decay=0.996):
        """
        Updates the target encoder weights via EMA of the context encoder.
        """
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1.0 - decay)
            
    def forward_context(self, board):
        """
        Encodes the current board state using the context encoder.
        """
        return self.context_encoder(board)
        
    def forward_target(self, next_board):
        """
        Encodes the next board state using the target encoder (no grads).
        """
        with torch.no_grad():
            return self.target_encoder(next_board)
        
    def forward_predict(self, context_latent, actions=None):
        """
        Predicts the target representation from the context latent and the action.
        """
        return self.predictor(context_latent, actions)
        
    def predict_value(self, context_latent):
        """
        Predicts the game outcome from the context representation.
        Returns value in [-1, 1]: +1 = white wins, -1 = black wins.
        """
        return self.value_head(context_latent)
