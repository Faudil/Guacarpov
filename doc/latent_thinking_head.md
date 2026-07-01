# Hybrid Latent Thinking Head (System-2 Reasoning)

The **Hybrid Latent Thinking Head** (`HybridLatentThinkingHead`) is an advanced policy architecture designed for the Guacarpov Chess Engine. 

Traditionally, chess engines like AlphaZero use a "System-1" neural network (a fast, single-pass ResNet/Transformer) to generate policy priors and value estimations, followed by a heavy explicit Monte Carlo Tree Search (MCTS) to perform "System-2" reasoning. 

To achieve high performance on mobile devices with strict latency budgets, the Latent Thinking Head attempts to internalize System-2 reasoning entirely within the continuous latent space. It replaces explicit tree search with a recurrent, adaptive pondering loop that operates on dense matrix multiplications.

---

## 1. Core Concepts

### Adaptive Computation Time (ACT)
Standard Neural Networks spend the exact same amount of compute evaluating a forced recapture as they do evaluating a highly complex, chaotic middlegame. This is wildly inefficient.
The Latent Thinking Head uses an **ACT Loop**. It recurrently refines its thought vector, outputting a dynamic "Halt Probability" at each step. If the position is obvious, it halts at step 1. If the position is complex, it continues to ponder up to a maximum number of steps (e.g., 10).

### The Ponder Penalty
To prevent the model from always using its maximum thinking budget, we apply a mathematical **Ponder Penalty**. The sum of the halt probabilities during the thinking loop is tracked. During RL and SFT training, a scaled penalty (e.g., `0.01 * mean_ponder_cost`) is added directly to the loss function. The model naturally learns to minimize its thinking time unless the extra compute yields a significantly better chess move that offsets the penalty.

### Value-Guided Halting
An engine should ponder when a position is critical or full of tactical landmines. To achieve this, the ACT cell natively queries the frozen JEPA `Value Head` at every step of its internal monologue. 
By concatenating the predicted Value scalar (e.g., `+0.8`) to its internal thought state, the cell mathematically "feels" the evaluation. If the Value prediction fluctuates wildly between thought steps, the cell learns to keep its halt probability low and continue searching. If the Value stabilizes, it halts.

### Gated Correction (Inner Monologue Take-backs)
If a model makes a bad assumption at step 1 of its thought process, an additive accumulator would permanently bake that mistake into the final action. 
To fix this, the Latent Thinking Head uses an LSTM-style **Forget Gate** (`update_gate`). If step 4 discovers a tactic that invalidates the evaluation from step 1, the cell can output a negative gate weight to effectively erase the initial blunder from its accumulated memory vector.

---

## 2. Architecture & Implementation Flow

The implementation is located in `JEPA_Policy/heads/latent_thinker_head.py`. 

### The Rollout Phase
When a board latent is passed to the head, it first queries the JEPA `TrajectoryPredictor`:
```python
future_latents = self.jepa_model.forward_predict(latent)
trajectory = torch.cat([latent.unsqueeze(1), future_latents], dim=1)
```
This generates an unconditional hallucinated future sequence of the game.

### The ACT Loop
The initial thought state is set to the current board latent. The loop begins:
1. **Query Value:** `value_pred = jepa_model.predict_value(thought_state)`
2. **Cross Attention:** The `ACTCell` uses Multi-Head Attention, treating the `thought_state` as the Query, and the hallucinated `trajectory` as the Keys/Values.
3. **Halting & Gating:** The cell projects the concatenated `[thought, value]` vector to a `halt_prob`, and projects the thought vector to an `update_gate`.
4. **State Accumulation:** 
   ```python
   accumulated_thought = (1 - update_gate) * accumulated_thought + update_gate * p_n * thought_state
   ```
5. **Early Exit:** If the cumulative halt probabilities reach `1.0`, the loop breaks early.

### Action Projection
Once the loop halts, the `accumulated_thought` vector is passed through a final linear projection to yield the `4672`-dimensional Leela Chess Zero action logits.

---

## 3. Rationale vs Explicit MCTS

While MCTS provides hard mathematical guarantees by traversing actual board states, it requires moving pieces, computing legal moves, and managing tree nodes in memory. This is highly expensive in Python and extremely draining on mobile device batteries.

The Latent Thinking Head runs purely on dense tensor operations (Matrix Multiplications and Attention). Modern GPUs and Neural Processing Units (NPUs) on mobile phones can execute a 10-step latent pondering loop in less than 5 milliseconds. By training the network to internalize its search using PPO (Proximal Policy Optimization) against itself and Stockfish, Guacarpov can achieve deep tactical vision with a sub-2-million parameter policy head.
