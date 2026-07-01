# Data Optimizations & Parallelization

Training an AlphaZero-style model on millions of chess games requires processing billions of board states. Storing these states naively as `111x8x8` float32 tensors would consume over **28 Kilobytes per state**, leading to Terabytes of required storage and crushing disk I/O bottlenecks during training.

To solve this, Guacarpov uses an extremely aggressive data compression and lazy-loading pipeline.

## 1. The Compact State Representation (~69 Bytes)

Instead of saving fully expanded tensors to disk, Guacarpov parses PGNs and compresses each board state into a custom structured Numpy array (`compact_state_dtype`) defined in `data_utils.py`:

```python
compact_state_dtype = np.dtype([
    ('board', np.uint8, 64),     # 64 bytes: 0-12 representing piece types
    ('castling', np.uint8),      # 1 byte: bitmask for castling rights
    ('en_passant', np.int8),     # 1 byte: -1 if none, or 0-63 square index
    ('halfmove', np.uint8),      # 1 byte: 50-move rule counter
    ('repetition', np.uint8),    # 1 byte: repetition counter
    ('turn', np.uint8),          # 1 byte: 1 for white, 0 for black
])
```
This reduces the storage requirement from **28,000 bytes down to just 69 bytes per position**—a >400x compression ratio. This allows us to store millions of games in small `.npz` chunk files that easily fit in RAM.

## 2. Bitwise Operations for Castling Rights

To save space and optimize extraction, all four castling rights are packed into a single 8-bit integer (`uint8`) using bit masking:
*   `0001` (1): White Kingside
*   `0010` (2): White Queenside
*   `0100` (4): Black Kingside
*   `1000` (8): Black Queenside

During training, `build_111_batch` extracts these channels instantly using vectorized bitwise `&` operators across the batch:
```python
c_w_k = ((castling & 1) > 0).float()
c_w_q = ((castling & 2) > 0).float()
# ...
```

## 3. Lazy "On-The-Fly" Tensor Reconstruction

Instead of loading massive tensors from disk, the DataLoader reads the compressed `np.uint8` structs and expands them into the full 111-channel float32 tensors *on the fly* just milliseconds before they are fed to the GPU.

*   **Vectorized One-Hot Encoding:** It uses `F.one_hot` to instantly blow up the 64-element piece array into a 12-channel spatial tensor.
*   **Vectorized History Lookup:** To build the 8-turn historical context, it calculates backward indices using `np.maximum(current_index - history_step, game_start_index)`. This guarantees that if we are at move 3, history states 4-8 safely duplicate move 1 without crossing into a different game.

## 4. Dataloader Multiprocessing (Parallelization)

To ensure the complex "on-the-fly" tensor reconstruction doesn't bottleneck the GPU, Guacarpov uses PyTorch's `IterableDataset` with heavy multiprocessing (`--num_workers 12`).

Each worker is assigned an isolated subset of the compressed `.npz` chunk files:
```python
worker_info = torch.utils.data.get_worker_info()
chunks = self.chunk_files[worker_info.id::worker_info.num_workers]
```
This allows 12 CPU cores to simultaneously decompress, bit-shift, one-hot encode, and build 111-channel tensors in parallel, ensuring the RTX 5070 Ti / Cloud GPU is constantly fed with data and never drops to 0% utilization.
