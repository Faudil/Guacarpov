# Guacarpov
A neural network chess engine that learns to play through Joint Embedding Predictive Architecture (JEPA), Latent Thinking (ACT), and Reinforcement Learning via Self-Play.

## Architecture: MoE-Latent Thinking Head

Guacarpov uses a **V-JEPA** style spatiotemporal context encoder connected to a **Latent Thinking Mixture of Experts (MoE)** policy head. 

Instead of relying on explicit Monte Carlo Tree Search (MCTS) which is computationally expensive for mobile devices, the network internalizes its search. It uses an **Adaptive Computation Time (ACT)** loop to mathematically ponder the board state, querying its own predicted value, and allowing it to self-correct thoughts before routing the final strategy to specialized experts for execution.

### Neural Data Flow Diagram

```mermaid
graph TD
    %% Base Inputs
    Board[Raw Board State] --> |Conv/Transformers| JEPA_CE(JEPA Context Encoder)
    JEPA_CE --> Latent[Latent State B, 256]
    
    %% The Pondering Loop (System-2)
    subgraph "Phase 1: Latent Thinking (ACT Loop)"
        Latent --> TrajPred{JEPA Trajectory Predictor}
        TrajPred --> |Hallucinated Futures| Sequence[Trajectory Sequence B, T, 256]
        
        Latent --> ThoughtState(Current Thought B, 256)
        
        ThoughtState --> ValHead{JEPA Value Head}
        ValHead --> ValPred(Predicted Value B, 1)
        
        ThoughtState --> CrossAttn((Cross Attention))
        Sequence --> CrossAttn
        
        CrossAttn --> NewThought(Next Thought State)
        
        NewThought --> HaltProj[Halt Projection]
        ValPred --> HaltProj
        HaltProj --> HaltProb((Halt Prob p_n))
        
        NewThought --> GateProj[Forget Gate Projection]
        GateProj --> Gate((Update Gate))
        
        Gate --> Accumulate[Accumulated Thought Vector]
        NewThought --> Accumulate
        HaltProb --> Accumulate
        
        %% Loop back
        Accumulate -.-> |Loop up to 10 steps| ThoughtState
    end
    
    %% The Execution Phase (System-1)
    subgraph "Phase 2: Sparse Expert Execution (MoE)"
        Accumulate --> GatingNet{MoE Gating Network}
        
        GatingNet --> |Top-k Routing| Expert1[Expert 1: Endgame]
        GatingNet --> |Top-k Routing| Expert2[Expert 2: Tactical]
        GatingNet --> Expert3[Expert 3: Positional]
        GatingNet --> Expert8[Expert 8: Openings]
        
        Expert1 --> OutputSum((Weighted Sum))
        Expert2 --> OutputSum
    end
    
    OutputSum --> Logits[Final 4672 Action Logits]
    
    %% Loss Functions
    HaltProb -.-> |Mean * Coeff| PonderLoss(Ponder Penalty Loss)
    GatingNet -.-> |Distribution| BalanceLoss(Load-Balancing Loss)
    PonderLoss --> UnifiedLoss((Unified Aux Loss))
    BalanceLoss --> UnifiedLoss
```

*For a detailed explanation of the architecture, see [doc/moe_latent_thinking_head.md](doc/moe_latent_thinking_head.md).*
