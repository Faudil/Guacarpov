# ConvNeXt JEPA Architecture

The ConvNeXt JEPA architecture represents a modernized convolutional backbone that bridges the gap between traditional Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). It incorporates design choices inspired by ViTs (such as large kernel sizes and inverted bottlenecks) while retaining the structural efficiency and local inductive biases of convolutions.

---

## 1. Architectural Details

The ConvNeXt JEPA model is defined in [`JEPA/jepa_convnext.py`](file:///home/faudil/Projects/python/Guacarpov/JEPA/jepa_convnext.py).

### ConvNeXt Block (`ConvNeXtBlock`)
Unlike classic ResNet blocks, the ConvNeXt block uses a layout inspired by the Vision Transformer's MLP block:
1.  **Depthwise 2D Convolution:** A `7x7` depthwise convolution (`groups=dim`) with padding of 3. This separates spatial filtering (mixing information across spatial dimensions) from channel mixing.
2.  **Permutation & LayerNorm:** The tensor is permuted from `(N, C, H, W)` to `(N, H, W, C)` to apply `LayerNorm` across the channels, replacing standard `BatchNorm2d`.
3.  **Inverted Bottleneck (Pointwise MLP):**
    - `Linear` layer projecting from `dim` to `4 * dim`.
    - `GELU` activation (replacing `ReLU`).
    - `Linear` layer projecting back to `dim`.
4.  **Permutation & Residual Add:** Permutes back to `(N, C, H, W)` and adds the residual input.

### ConvNeXt Encoder (`ConvNeXtEncoder`)
- **Stem:** Projects raw input channels (`12` or `111`) to the intermediate block dimension (`dim`) using a `3x3` convolution with `BatchNorm2d` and `GELU`.
- **Block Stack:** Iterates through `num_blocks` of `ConvNeXtBlock`s.
- **Normalization:** Normalizes the final feature grids using `LayerNorm`.
- **Projection Head:**
  - `Flatten` (collapses `dim * 8 * 8` to a flat vector).
  - `Linear` projection to `1024` units.
  - `GELU`.
  - `Linear` projection to the target `latent_dim`.

### Predictor (`Predictor`)
- A deeper 4-layer MLP that maps context latents to predicted future target representations:
  - `Linear` -> `LayerNorm` -> `GELU` -> `Linear` -> `LayerNorm` -> `GELU` -> `Linear` -> `LayerNorm` -> `GELU` -> `Linear`.

### Value Head
- A 2-layer MLP with `GELU` activation:
  - `Linear` -> `GELU` -> `Linear` -> `Tanh`.

---

## 2. Strengths

- **Large Receptive Field:** The `7x7` depthwise convolutions allow each block to see a wider area of the board early in the network, helping the model capture long-range piece interactions (such as diagonal pins and rook files) much faster than standard ResNet blocks.
- **Separation of Space and Channel Mixing:** By decoupling spatial mixing (depthwise convolution) from channel mixing (pointwise linear layers), the network represents complex chess board patterns with significantly fewer parameters and operations.
- **Modern Normalization & Activation:** Replacing `BatchNorm2d` and `ReLU` with `LayerNorm` and `GELU` stabilizes training and matches modern deep learning standards, making the model ideal for large-scale training.
- **CNN Efficiency:** Retains the high efficiency, fast GPU inference, and translation/spatial inductive bias of CNNs without requiring patch-segmentation.

---

## 3. Weaknesses

- **Vulnerable to Overfitting on Small Data:** The increased capacity and expressive power of ConvNeXt blocks make them more prone to overfitting compared to simple ResNets if trained on small databases.
- **Translation Invariance Constraints:** Like all CNNs, it natively assumes spatial translation invariance, which doesn't perfectly model chess (where specific squares like e4/d4 or king castles have unique properties).

---

## 4. Rationale & Design Choice

ConvNeXt was introduced as a modern, high-throughput alternative to the Vision Transformer. In Guacarpov, it acts as the default scaling architecture for reinforcement learning and policy distillation, providing transformer-like representation capacity while maintaining the raw training speed and hardware optimization of a pure convolutional network.

---

## 5. Configuration Arguments

The ConvNeXt model can be customized via the CLI using:

| CLI Parameter | Default | Description |
|---|---|---|
| `--arch` | `convnext` | Specifies the backbone architecture. |
| `--in_channels` | `12` / `111` | Channels of input representation. |
| `--latent_dim` | `512` | Continuous latent representation size. |
| `--num_res_blocks` | `16` | Mapped to `num_blocks` of ConvNeXt blocks. |
| `--num_filters` | `256` | Mapped to the intermediate channel dimension `dim`. |
