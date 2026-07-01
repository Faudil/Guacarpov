# Step 3: Reinforcement Learning (RL)

The final phase of Guacarpov's training pipeline is Reinforcement Learning. While Supervised Fine-Tuning (SFT) teaches the model to imitate human players, RL allows the model to surpass human limitations by discovering new strategies through self-play and adversarial training.

## The Objective

The goal of RL is to optimize the policy (move probabilities) to maximize the expected reward (winning the game). We rely on the rich latent representations learned during JEPA pre-training and the baseline chess understanding acquired during SFT.

## Training Regimes

To prevent "echo chamber" collapse (where a model only learns to beat its own specific weaknesses), Guacarpov employs an asymmetric and diverse training regime:

1.  **Self-Play:** The active RL model plays against older versions of itself. This provides a curriculum that scales in difficulty as the model improves.
2.  **SFT Baselines:** The RL model regularly plays against the frozen SFT model to ensure it doesn't "forget" fundamental chess principles while exploring exotic strategies.
3.  **Stockfish Benchmarking:** The model plays against Stockfish (at various difficulty levels) to force generalization against high-quality, non-neural engine moves.

## The Advantage Function

The RL algorithm uses the JEPA **Value Head** to compute the *Advantage* of a move. 
*   If a move leads to a position with a higher value than expected, the policy logit for that move is strengthened.
*   If a move leads to a worse position, it is penalized.

**Design Choice:** During RL, only the RL agent learns from its decisions. When it plays against Stockfish or the SFT baseline, we enforce selective advantage computation so the RL model isn't penalized for the opponent's blunders, ensuring clean gradient updates.
