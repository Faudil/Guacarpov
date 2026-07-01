# ResNet JEPA Architecture

The ResNet (Residual Network) architecture represents the classic convolutional backbone of the Guacarpov Chess JEPA engine. It serves as the baseline, leveraging spatial inductive biases standard in chess representation models.

---

## 1. Architectural Details

The ResNet JEPA model is defined in [`JEPA/jepa_model.py`](file:///home/faudil/Projects/python/Guacarpov/JEPA/jepa_model.py).

### Context & Target Encoder (`ChessEncoder`)
- **Input Convolution:** A `3x3` 2D Convolution with padding of 1 to project the raw chess board channels (`12` or `111`) to `num_filters` channels, followed by `BatchNorm2d` and a `ReLU` activation.
- **Residual Towers:** A series of `num_res_blocks` standard residual blocks.
  - Each `ResidualBlock` contains two successive `3x3` convolutions with padding, batch normalizations, a shortcut skip connection, and `ReLU` activations.
- **Latent Projection Head:**
  - `Flatten` (collapses `num_filters * 8 * 8` to a flat vector).
  - `Linear` projection to `512` units.
  - `LayerNorm` + `ReLU`.
  - `Linear` projection to the target `latent_dim` (default `256` or `512`).
  - `LayerNorm` (normalizes the final continuous latent vector).

### Predictor (`Predictor`)
- An MLP that takes the current context latent and predicts the target representation.
- Structured as: `Linear` -> `LayerNorm` -> `GELU` -> `Linear` -> `LayerNorm` -> `GELU` -> `Linear`.

### Value Head
- A simple MLP mapping the context latent to a scalar value prediction in the range `[-1, 1]`:
  - `Linear` -> `ReLU` -> `Linear` -> `Tanh`.

---

## 2. Strengths

- **Strong Spatial Inductive Bias:** 2D Convolutions naturally capture localized spatial patterns (e.g., pawn structures, king safety, knight forks) that exist within small neighborhoods of chess squares.
- **Training Stability:** The skip connections in residual blocks allow gradients to flow easily during backpropagation, making training highly stable.
- **High Computational Efficiency:** Small `3x3` kernels are heavily optimized on modern GPUs, making training and inference extremely fast.
- **Low Parameter Count:** Highly lightweight compared to transformer-based alternatives, reducing the risk of overfitting on smaller datasets.

---

## 3. Weaknesses

- **Limited Receptive Field per Layer:** A single `3x3` convolutional layer can only see adjacent squares. Capturing long-range piece interactions (e.g., a queen pinning a knight across the entire board) requires passing through multiple stacked layers.
- **Translation Invariance Assumption:** Chess is not translation invariant. A rook on the first rank behaves differently from a rook on the fourth rank. Standard convolutions apply the same weights across all ranks and files, which must be overridden by learning position-dependent piece dynamics.
- **Lack of Temporal Modeling:** Standard ResNet processes a static snapshot of the board (or requires stack-channel representations like the 111-channel format to approximate history).

---

## 4. Rationale & Design Choice

The ResNet encoder backbone is chosen as a standard baseline because it matches the architecture family popularized by **AlphaZero** and **Leela Chess Zero**. It provides a robust, fast-to-train baseline for policy and value distillation and serves as the benchmark against which newer architectures are measured.

---

## 5. Configuration Arguments

The ResNet model can be customized via the CLI using:

| CLI Parameter | Default | Description |
|---|---|---|
| `--arch` | `resnet` | Specifies the backbone architecture. |
| `--in_channels` | `12` | Channels of input representation (`12` for v2, `111` for v3). |
| `--latent_dim` | `256` | Continuous latent representation size. |
| `--num_res_blocks` | `4` | Number of residual blocks in the encoder tower. |
| `--num_filters` | `64` | Intermediate feature channel depth. |
