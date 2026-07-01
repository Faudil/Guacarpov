# Step 2: Supervised Fine-Tuning (SFT)

Once the JEPA model has learned a rich, structured latent space (`JEPA/train.py`), the next step is to teach the model how to actually play chess using that knowledge (`JEPA_Policy/train_sft.py`).

We do this via Supervised Fine-Tuning (SFT) on a dataset of high-Elo human games (e.g., from Lichess).

## The Action Space

Guacarpov uses the action space from **Leela Chess Zero (LCZero)**. Instead of outputting a simple string like "e2e4", the model outputs a 4672-dimensional vector. Each index in this vector corresponds to a highly specific geometric transformation (e.g., "move a knight two squares up and one square right", or "promote a pawn to a queen on the c-file").

## Policy Head Choices

To map the 256d JEPA latent vector to the 4672-dim action space, Guacarpov provides several Policy Head architectures:

1.  **Linear (`nn.Linear`):** The simplest head. Fast, but generally lacks the non-linear capacity to decode complex piece interactions into 4672 distinct classes.
2.  **MLP (Multi-Layer Perceptron):** The recommended default. Uses hidden layers and GELU activations to provide the non-linear capacity required for accurate move prediction.
3.  **Transformer:** Uses self-attention layers on the latent space before projecting to the action space.
4.  **MoE (Mixture of Experts):** Routes the latent representation to specialized "expert" sub-networks based on the board context, useful for separating opening theory from endgame tactics.

## Validation Strategy (No Legal Masking)

During the SFT training loop, we evaluate validation accuracy on the raw, unmasked logits across all 4672 actions.

**Design Choice:** We explicitly **do not** mask illegal moves during validation. 
*   **Why?** Calculating legal moves for massive batches during training requires significant CPU overhead and slows down training immensely. 
*   **The Benefit:** By forcing the model to pick from all 4672 moves (instead of just the ~30 legal ones), we rigorously test if the JEPA representations have truly internalized the physical rules of chess. High unmasked accuracy proves the model understands geometry without a rules engine holding its hand.

## Inference & Gameplay (`policy_model.py`)

When the model is deployed to actually play a game, we introduce programmatic guardrails:

1.  **Legal Move Masking:** The engine queries `python-chess` for the valid moves in the current position and applies a `-inf` mask to all illegal move logits. This guarantees the engine never plays an illegal move.
2.  **1-Step Lookahead:** Before finalizing a move, the engine looks one step into the future using the JEPA Value Head. If a move leads to an immediate Checkmate, it forces that move. If a move blunders into a Draw while winning, it penalizes it.
