#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def run_command(cmd):
    print(f"\nExecuting: {' '.join(cmd)}")
    try:
        # Run process and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"\n❌ Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user.")
        if 'process' in locals() and process:
            process.terminate()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the full Chess JEPA pipeline (SFT -> RL -> Benchmark).")
    
    # General Options
    parser.add_argument('--python_bin', type=str, default='venv/bin/python', help="Python executable to use.")
    parser.add_argument('--jepa_checkpoint', type=str, default='JEPA/chess_jepa.pth', help="Path to base pre-trained JEPA model.")
    parser.add_argument('--head_type', type=str, default='moe', choices=['linear', 'mlp', 'transformer', 'moe', 'latent_thinker', 'moe_latent_thinker'],
                        help="Policy head architecture: linear, mlp, transformer, moe, latent_thinker, or moe_latent_thinker.")
    parser.add_argument('--spatial_dim', type=int, default=128, help="ConvNeXt filter dimension for Spatiotemporal.")
    parser.add_argument('--spatial_blocks', type=int, default=4, help="ConvNeXt block count for Spatiotemporal.")
    parser.add_argument('--temporal_layers', type=int, default=4, help="Transformer layer count for Spatiotemporal.")
    parser.add_argument('--temporal_heads', type=int, default=8, help="Transformer head count for Spatiotemporal.")
    parser.add_argument('--stockfish_path', type=str, default='stockfish_bin/stockfish/stockfish-ubuntu-x86-64', help="Path to Stockfish binary.")
    
    # SFT Phase Options (Downloads games online)
    parser.add_argument('--skip_sft', action='store_true', help="Skip Supervised Fine-Tuning phase.")
    parser.add_argument('--sft_checkpoint', type=str, default='JEPA_Policy/chess_sft_policy.pth', help="Path to SFT checkpoint.")
    parser.add_argument('--sft_games', type=int, default=1000, help="Number of games to download/parse from Lichess.")
    parser.add_argument('--sft_min_elo', type=int, default=1600, help="Min Elo rating of players to collect from Lichess.")
    parser.add_argument('--sft_epochs', type=int, default=5, help="Number of SFT epochs.")
    parser.add_argument('--sft_batch_size', type=int, default=1024, help="Batch size for SFT training.")
    parser.add_argument('--sft_lr', type=float, default=1e-3, help="Learning rate for SFT.")
    
    # RL Phase Options (Self-Play against SFT, Stockfish, Random, Self)
    parser.add_argument('--skip_rl', action='store_true', help="Skip Reinforcement Learning phase.")
    parser.add_argument('--rl_checkpoint', type=str, default='JEPA_RL/chess_rl_policy.pth', help="Path to save RL checkpoint.")
    parser.add_argument('--rl_epochs', type=int, default=5, help="Number of RL training epochs.")
    parser.add_argument('--rl_games_per_epoch', type=int, default=200, help="Number of games generated per epoch.")
    parser.add_argument('--rl_batch_size', type=int, default=1024, help="Batch size for RL optimization.")
    parser.add_argument('--rl_lr', type=float, default=1e-5, help="Learning rate for RL.")
    parser.add_argument('--rl_workers', type=int, default=4, help="Number of parallel worker processes for self-play.")
    parser.add_argument('--opp_sft', type=float, default=0.5, help="RL opponent SFT probability.")
    parser.add_argument('--opp_stockfish', type=float, default=0.2, help="RL opponent Stockfish probability.")
    parser.add_argument('--opp_random', type=float, default=0.2, help="RL opponent Random probability.")
    parser.add_argument('--opp_self', type=float, default=0.1, help="RL opponent Self probability.")
    
    # Benchmarking Options
    parser.add_argument('--skip_bench', action='store_true', help="Skip benchmarking phase (SFT vs RL comparison).")
    parser.add_argument('--bench_games', type=int, default=50, help="Number of match games for benchmarking.")
    parser.add_argument('--bench_temp', type=float, default=0.5, help="Sampling temperature during benchmark matches.")
    
    args = parser.parse_args()
    
    # Make sure python executable exists
    if not os.path.exists(args.python_bin):
        print(f"⚠️ Warning: Python executable '{args.python_bin}' not found. Falling back to default 'python3'.")
        args.python_bin = 'python3'

    # Step 1: Supervised Fine-Tuning (SFT)
    if not args.skip_sft:
        print("\n=== STEP 1: SUPERVISED FINE-TUNING (SFT) ===")
        print(f"Downloading/parsing {args.sft_games} Lichess games...")
        sft_cmd = [
            args.python_bin, 'JEPA_Policy/train_sft.py',
            '--jepa_checkpoint', args.jepa_checkpoint,
            '--save_path', args.sft_checkpoint,
            '--num_games', str(args.sft_games),
            '--min_elo', str(args.sft_min_elo),
            '--epochs', str(args.sft_epochs),
            '--batch_size', str(args.sft_batch_size),
            '--lr', str(args.sft_lr),
            '--head_type', args.head_type
        ]
        run_command(sft_cmd)
    else:
        print("\n⏭️ Skipping Step 1 (SFT).")

    # Step 2: Reinforcement Learning (RL)
    if not args.skip_rl:
        print("\n=== STEP 2: REINFORCEMENT LEARNING (RL) ===")
        # Validate SFT checkpoint exists as baseline
        if not os.path.exists(args.sft_checkpoint):
            print(f"❌ Error: Baseline SFT checkpoint not found at '{args.sft_checkpoint}'. Please run SFT first.")
            sys.exit(1)
            
        rl_cmd = [
            args.python_bin, 'JEPA_RL/train_rl.py',
            '--jepa_checkpoint', args.jepa_checkpoint,
            '--policy_checkpoint', args.sft_checkpoint,
            '--save_path', args.rl_checkpoint,
            '--epochs', str(args.rl_epochs),
            '--games_per_epoch', str(args.rl_games_per_epoch),
            '--batch_size', str(args.rl_batch_size),
            '--lr', str(args.rl_lr),
            '--num_selfplay_workers', str(args.rl_workers),
            '--head_type', args.head_type,
            '--opp_sft', str(args.opp_sft),
            '--opp_stockfish', str(args.opp_stockfish),
            '--opp_random', str(args.opp_random),
            '--opp_self', str(args.opp_self),
            '--stockfish_path', args.stockfish_path
        ]
        run_command(rl_cmd)
    else:
        print("\n⏭️ Skipping Step 2 (RL).")

    # Step 3: Benchmarking (SFT vs RL)
    if not args.skip_bench:
        print("\n=== STEP 3: BENCHMARKING (SFT vs RL) ===")
        # Verify checkpoints
        if not os.path.exists(args.sft_checkpoint):
            print(f"❌ Error: SFT checkpoint '{args.sft_checkpoint}' not found. Cannot run benchmark.")
            sys.exit(1)
        if not os.path.exists(args.rl_checkpoint):
            print(f"❌ Error: RL checkpoint '{args.rl_checkpoint}' not found. Cannot run benchmark.")
            sys.exit(1)
            
        bench_cmd = [
            args.python_bin, 'JEPA_RL/compare_policies.py',
            '--sft_checkpoint', args.sft_checkpoint,
            '--rl_checkpoint', args.rl_checkpoint,
            '--num_games', str(args.bench_games),
            '--temperature', str(args.bench_temp)
        ]
        run_command(bench_cmd)
    else:
        print("\n⏭️ Skipping Step 3 (Benchmarking).")

    print("\n🎉 Pipeline Run Completed!")

if __name__ == '__main__':
    main()
