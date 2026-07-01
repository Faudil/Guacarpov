#!/bin/bash

# Ensure the script exits if a command fails
set -e

# Setup usage instructions
function show_usage {
    echo "Usage: ./vast_deploy.sh <VAST_IP> <VAST_PORT> [ARCHITECTURE] [FLAGS...]"
    exit 1
}

if [ "$#" -lt 2 ]; then
    show_usage
fi

VAST_IP=$1
VAST_PORT=$2
shift 2

ARCH="spatiotemporal"
if [[ $# -gt 0 && ! "$1" == --* ]]; then
    ARCH=$1
    shift
fi
EXTRA_ARGS="$@"

IS_TEST=false
if [[ " $EXTRA_ARGS " =~ " --test " ]]; then
    IS_TEST=true
fi

# Vast.ai SSH typically uses root and can have changing host keys
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $VAST_PORT"

echo "=================================================="
echo "🚀 Deploying to Vast.ai Instance: $VAST_IP:$VAST_PORT"
echo "Architecture: $ARCH"
echo "Test Mode: $IS_TEST"
echo "=================================================="

echo "[1/4] Creating workspace directories on remote instance..."
ssh $SSH_OPTS root@$VAST_IP "mkdir -p /workspace/Guacarpov"

echo "[2/4] Uploading all code directories and binaries..."
# We explicitly exclude .pt, .npz, and .pth caches/models from the code directories
rsync -avzP --exclude="*.pt" --exclude="*.npz" --exclude="*.pth" --exclude="__pycache__" -e "ssh $SSH_OPTS" ./JEPA/ root@$VAST_IP:/workspace/Guacarpov/JEPA/
rsync -avzP --exclude="*.pt" --exclude="*.npz" --exclude="*.pth" --exclude="__pycache__" -e "ssh $SSH_OPTS" ./JEPA_Policy/ root@$VAST_IP:/workspace/Guacarpov/JEPA_Policy/
rsync -avzP --exclude="*.pt" --exclude="*.npz" --exclude="*.pth" --exclude="__pycache__" -e "ssh $SSH_OPTS" ./JEPA_RL/ root@$VAST_IP:/workspace/Guacarpov/JEPA_RL/
rsync -avzP --exclude="__pycache__" -e "ssh $SSH_OPTS" ./stockfish_bin/ root@$VAST_IP:/workspace/Guacarpov/stockfish_bin/
rsync -avzP -e "ssh $SSH_OPTS" ./run_pipeline.py root@$VAST_IP:/workspace/Guacarpov/

echo "[3/4] Uploading V3 dataset (Streaming via tar to avoid massive file-overhead)..."
tar -cf - ./jepa_v3_data/ | ssh $SSH_OPTS root@$VAST_IP "cd /workspace/Guacarpov/ && tar -xf -"

echo "[4/4] Installing dependencies and launching training pipeline..."

CMD=""
# We strip --skip_jepa from the args passed to the scripts since it doesn't recognize it
CLEAN_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--skip_jepa//g')

if [[ ! " $EXTRA_ARGS " =~ " --skip_jepa " ]]; then
    if [ "$IS_TEST" = true ]; then
        CMD="python3 JEPA/train.py --arch $ARCH --data_dir jepa_v3_data --in_channels 111 --epochs 1 $CLEAN_ARGS && "
    else
        CMD="python3 JEPA/train.py --arch $ARCH --data_dir jepa_v3_data --in_channels 111 $CLEAN_ARGS && "
    fi
fi

if [ "$IS_TEST" = true ]; then
    CMD="${CMD}python3 run_pipeline.py --python_bin python3 --jepa_checkpoint JEPA/chess_jepa.pth --sft_games 10 --sft_epochs 1 --rl_epochs 1 --rl_games_per_epoch 10 --bench_games 2 $CLEAN_ARGS"
else
    CMD="${CMD}python3 run_pipeline.py --python_bin python3 --jepa_checkpoint JEPA/chess_jepa.pth $CLEAN_ARGS"
fi

ssh $SSH_OPTS root@$VAST_IP << EOF
    set -e
    cd /workspace/Guacarpov
    
    echo "Installing Python requirements..."
    # Do NOT pip install torch here, because the pytorch Docker image already has the correct CUDA 12.8 version built-in!
    pip install numpy tqdm python-chess requests
    
    echo "Making Stockfish executable..."
    chmod +x stockfish_bin/stockfish/stockfish-ubuntu-x86-64 || true
    
    echo "Starting full pipeline inside tmux..."
    # Group the entire command in { } so that tee captures output even if the first command crashes
    tmux new-session -d -s jepa_train "{ $CMD } 2>&1 | tee training.log"
    
    echo "--------------------------------------------------"
    echo "✅ Setup Complete & Pipeline Launched!"
    echo "To view your training logs, SSH into the machine and run:"
    echo "  tmux attach-session -t jepa_train"
    echo "Or just view the log file:"
    echo "  tail -f /workspace/Guacarpov/training.log"
EOF
