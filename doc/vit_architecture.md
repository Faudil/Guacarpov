# Vision Transformer (ViT) JEPA Architecture

The Vision Transformer (ViT) JEPA architecture completely abandons traditional convolutions. It treats the chess board as a set of 64 discrete tokens (representing individual squares) and uses global self-attention to model long-range relationships between pieces.

---

## 1. Architectural Details

The ViT JEPA model is defined in [`JEPA/jepa_vit.py`](file:///home/faudil/Projects/python/Guacarpov/JEPA/jepa_vit.py).

### Square-Patch Tokenization
- **Board Tokenization:** The `8x8` chess board is treated as a sequence of `64` spatial patches, where each patch is a single square.
- **Linear Projection:** A linear layer (`self.square_proj`) projects the raw channels of each square (`12` or `111`) into a continuous token embedding of size `d_model`.
- **Classification Token (`CLS`):** A learnable `[CLS]` token is prepended to the sequence of 64 square tokens. The final representation of the `[CLS]` token acts as the global state summary.
- **Positional Embeddings:** Learnable 2D positional embeddings of size `(1, 65, d_model)` are added to all tokens to preserve rank and file coordinate awareness.

### Transformer Encoder Layers (`TransformerEncoderLayerCustom`)
The model uses a **Pre-Layer Normalization (Pre-LN)** configuration, which is essential for stabilizing deep transformer training:
1.  **Multi-Head Self-Attention:** Each token attends to all other 64 tokens, enabling the model to dynamically compute relations between any two arbitrary squares.
2.  **Feed-Forward Network (FFN):** An inverted bottleneck MLP containing:
    - `Linear` projecting `d_model` to `4 * d_model`.
    - `GELU` activation.
    - `Linear` projecting back to `d_model`.
3.  **Residual & Norm Connections:** Standard residual skip connections bypass the attention and FFN blocks.

### Global Latent Projection
- The output corresponding to the `[CLS]` token is isolated, passed through a final `LayerNorm`, and projected via a linear layer to `latent_dim` (default `512`).

### Predictor & Value Head
- **Predictor:** An MLP with layers: `Linear` -> `LayerNorm` -> `GELU` -> `Linear` -> `LayerNorm` -> `GELU` -> `Linear`.
- **Value Head:** A simple MLP mapping the CLS latent to a scalar value prediction: `Linear` -> `GELU` -> `Linear` -> `Tanh`.

---

## 2. Strengths

- **True Global Receptive Field:** Global self-attention allows the model to compute interactions between any two squares on the board in a single step (layer). It perfectly represents long-range diagonal bishops, sliding rooks, and king-castle relationships.
- **No Translation Invariance Assumptions:** Positional embeddings allow the model to learn rank and file specific properties natively (e.g., that e4 is a center square, and a pawn on the 7th rank is about to promote).
- **Infinite Scalability:** ViT models scale extremely well. As dataset sizes and model parameter counts increase, ViTs consistently outperform convolutional networks.

---

## 3. Weaknesses

- **High Data Requirement:** ViT models lack the built-in inductive biases of convolutions (like locality). Consequently, they require significantly more training data (or longer pre-training) to learn basic board geometries.
- **High Computational Overhead:** The cost of self-attention scales quadratically ($O(N^2)$) with the sequence length. While $N=64$ is small, training deep ViTs requires substantially more VRAM and compute compared to ResNets.
- **Slower Convergence:** Initially, the loss decreases slower compared to ResNets because the model must learn how to "patch" the board and represent distance from scratch.

---

## 4. Rationale & Design Choice

The Vision Transformer JEPA model is designed for long-term scalability. When training on massive datasets (e.g., hundreds of millions of positions from Lichess), ViT backbones are capable of learning deep strategic representations that surpass the rigid structures of classical CNNs.

---

## 5. Configuration Arguments

The ViT model can be customized via the CLI using:

| CLI Parameter | Default | Description |
|---|---|---|
| `--arch` | `vit` | Specifies the backbone architecture. |
| `--in_channels` | `111` | Channels of input representation. |
| `--latent_dim` | `512` | Continuous latent representation size. |
| `--num_res_blocks` | `8` | Mapped to `num_layers` (transformer layers). |
| `--num_filters` | `512` | Mapped to `d_model` (embedding dimension). |
