# Spatiotemporal JEPA Architecture

The Spatiotemporal JEPA architecture is inspired by modern video prediction models (such as V-JEPA). Instead of treating a chess board as a single static snapshot, it models chess as a temporal trajectory (a sequence of moves and board states over time), learning representations of *dynamics and momentum* rather than static piece geometry.

---

## 1. Architectural Details

The Spatiotemporal JEPA model is defined in [`JEPA/jepa_spatiotemporal.py`](file:///home/faudil/Projects/python/Guacarpov/JEPA/jepa_spatiotemporal.py).

### Spatial Encoder (`SpatialEncoder`)
- Used to extract spatial latent vectors from individual board frames.
- **Stem:** Convolves input channels (`15` default per frame) to the intermediate size `dim` via a `3x3` convolution with `BatchNorm2d` and `GELU`.
- **Blocks:** Passes the spatial grid through `num_blocks` of `ConvNeXtBlock`s.
- **Head:** Flattens and projects the final spatial representation to the model's sequence dimension (`d_model`) with a `LayerNorm` output.

### Context Encoder (`SpatioTemporalEncoder`)
Aggregates a sequence of past board states (`T_history` frames) into a single trajectory context latent:
1.  **Frame Encoding:** Passes all `T_history` frames through the `SpatialEncoder` in parallel, yielding a sequence of shape `(B, T_history, d_model)`.
2.  **Temporal Positional Embeddings:** Adds learned positional embeddings to preserve temporal order (which frame came first).
3.  **CLS Token aggregation:** Prepends a learnable `[CLS]` token to the sequence.
4.  **Temporal Transformer Encoder:** Processes the sequence of length `T_history + 1` through a standard `nn.TransformerEncoder` with `num_layers`.
5.  **Output:** Returns the normalized `[CLS]` token latent as the aggregated history context.

### Target Encoder (`SpatialEncoder`)
- **Asymmetric Structure:** Crucially, the target encoder does *not* contain a temporal transformer. It only needs the `SpatialEncoder` sub-network to map ground truth future boards (`T_future` frames) to their spatial targets.
- **Weights:** Updated via Exponential Moving Average (EMA) of the context encoder's spatial sub-network.

### Trajectory Predictor (`TrajectoryPredictor`)
- A transformer decoder that predicts multiple future steps (latents) from the historical context:
  - Takes the context representation as keys and values (memory).
  - Takes learnable query embeddings of shape `(1, t_future, d_model)` as inputs (queries).
  - Decodes the queries through `nn.TransformerDecoder` layers into predicted target embeddings for the next `T_future` board states.

### Value Head
- A 3-layer MLP with `LayerNorm` and `GELU`:
  - `Linear` -> `LayerNorm` -> `GELU` -> `Linear` -> `LayerNorm` -> `GELU` -> `Linear` -> `Tanh`.

---

## 2. Strengths

- **Understands Momentum and Trajectories:** By processing a sequence of boards, the model naturally encodes information like tactical threats, piece trajectories, attacking momentum, and structural changes over time.
- **Multi-Step Future Prediction:** Rather than predicting just 1 step ahead, the trajectory predictor can model multiple future states (`T_future`), making it highly aligned with deep multi-step search algorithms (lookahead/MCTS).
- **Asymmetric Representation Learning:** By keeping the target encoder purely spatial (no temporal mixing) and training the context encoder to predict future spatial states, the model learns a clean decomposition of space and time.

---

## 3. Weaknesses

- **Action-Blindness (No Move Conditioning):** The predictor attempts to infer the future state latent purely from the historical context *without knowing which move will actually be played*. This forces the model to predict the "average" or "most likely" next state (marginalizing over all legal moves), which limits its utility as an exact transition dynamics model (World Model) for algorithms like MCTS that require precise $S_{t+1} = f(S_t, A_t)$ rollouts.
- **High Memory Footprint:** Storing sequences of boards and intermediate activations across both spatial and temporal dimensions significantly increases VRAM usage.
- **High Training Complexity:** Requires custom data-loading structures (trajectory chunks) and is more sensitive to hyperparameters (e.g. temporal vs. spatial loss balance).
- **Inference Latency:** Evaluating sequential states via spatial-then-temporal passes is slower than processing a single pre-stacked snapshot.

---

## 4. Rationale & Design Choice

The Spatiotemporal JEPA model is the most advanced representation learning architecture in the Guacarpov project. It addresses the core limitation of standard static policy models by allowing the agent to read the "tempo" and history of the game, mirroring how human players analyze strategic changes and positional flows across the game's timeline.

---

## 5. Configuration Arguments

The Spatiotemporal model can be customized via the CLI using:

| CLI Parameter | Default | Description |
|---|---|---|
| `--arch` | `spatiotemporal` | Specifies the backbone architecture. |
| `--latent_dim` | `512` | Mapped to `d_model` (sequence representation size). |
| `--spatial_dim` | `128` | Filter dimension for the spatial ConvNeXt stem. |
| `--spatial_blocks` | `4` | ConvNeXt block count in the spatial sub-network. |
| `--temporal_layers` | `4` | Layer count in the temporal transformer. |
| `--temporal_heads` | `8` | Head count in the temporal transformer. |
| `--in_channels` | `15` | Default input channels per frame. |
