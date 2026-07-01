# Architectures & Data Representation

Guacarpov processes chess board states as multi-channel 2D spatial tensors (similar to image processing in computer vision). 

## Data Representation

The engine supports two input formats:

1.  **12-Channel (Legacy v2):** A basic `(12, 8, 8)` representation where each channel represents the presence of a specific piece type (Pawn, Knight, Bishop, Rook, Queen, King) for a specific color (White, Black).
2.  **111-Channel (v3 AlphaZero Style):** A highly complex `(111, 8, 8)` representation that provides a complete Markov state of the game. It includes:
    *   **Historical Boards:** The last 8 board states (to capture piece trajectories and momentum).
    *   **Game Context:** Castling rights, en passant squares, half-move clock (for the 50-move rule), and repetition counters.

## Encoder Backbones

To compress these spatial grids into a rich 256-dimensional continuous latent vector, Guacarpov supports four interchangeable encoder architectures (`JEPA/factory.py`):

1.  **[ResNet](resnet_architecture.md) (`jepa_model.py`):** The standard workhorse. Uses 2D convolutions with skip connections. Fast to train, highly effective for spatial data, and excellent for local piece relationships.
2.  **[ConvNeXt](convnext_architecture.md) (`jepa_convnext.py`):** A modernized convolutional architecture that uses depthwise convolutions and larger kernel sizes (7x7) to mimic Vision Transformer performance while retaining CNN efficiency.
3.  **[ViT (Vision Transformer)](vit_architecture.md) (`jepa_vit.py`):** Treats the 8x8 chess board as a sequence of patches. Uses global self-attention to understand long-range piece interactions (e.g., a Bishop on a1 pinning a Queen on h8). Harder to train but scales well.
4.  **[SpatioTemporal](spatiotemporal_architecture.md) (`jepa_spatiotemporal.py`):** Designed specifically to process a time-series of board states, learning representations not just of the board, but of *movement over time*.
