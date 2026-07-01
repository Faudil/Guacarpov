import torch

from jepa_model import ChessJEPA
from jepa_convnext import ChessJEPA_ConvNeXt
from jepa_vit import ChessJEPA_ViT
from jepa_spatiotemporal import ChessJEPA_SpatioTemporal

    
def build_jepa_model(args):
    """
    Instantiates the correct JEPA architecture based on args.
    """
    arch = getattr(args, 'arch', 'resnet')
    in_channels = getattr(args, 'in_channels', 12)
    latent_dim = getattr(args, 'latent_dim', 256)
    num_res_blocks = getattr(args, 'num_res_blocks', 4)
    num_filters = getattr(args, 'num_filters', 64)
    
    print(f"Detected JEPA Architecture: {arch.upper()} (in_channels={in_channels}, latent_dim={latent_dim})")
    
    if arch == 'resnet':
        return ChessJEPA(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters
        )
    elif arch == 'convnext':
        return ChessJEPA_ConvNeXt(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_blocks=num_res_blocks,
            dim=num_filters
        )
    elif arch == 'vit':
        return ChessJEPA_ViT(
            in_channels=in_channels,
            latent_dim=latent_dim,
            d_model=num_filters,
            nhead=8,
            num_layers=num_res_blocks
        )
    elif arch == 'spatiotemporal':
        spatial_dim = getattr(args, 'spatial_dim', 128)
        spatial_blocks = getattr(args, 'spatial_blocks', 4)
        temporal_layers = getattr(args, 'temporal_layers', 4)
        temporal_heads = getattr(args, 'temporal_heads', 8)
        
        return ChessJEPA_SpatioTemporal(
            in_channels=in_channels,
            d_model=latent_dim,
            t_history=1,
            t_future=1,
            dim=spatial_dim,
            num_blocks=spatial_blocks,
            nhead=temporal_heads,
            num_layers=temporal_layers
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def load_jepa_from_checkpoint(checkpoint_path, device):
    """
    Dynamically instantiates and loads the correct JEPA architecture based on the saved arguments.
    Supports backward compatibility with older ResNet v2 checkpoints.
    """
    print(f"Loading JEPA model from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract args, falling back to defaults for very old v2 checkpoints
    args = checkpoint.get('args', type('', (), {})()) 
    
    model = build_jepa_model(args)
        
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ JEPA model successfully loaded.")
    return model, args
