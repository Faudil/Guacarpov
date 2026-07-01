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

NUM_CHUNKS="all"
if [[ " $EXTRA_ARGS " =~ " --upload_chunks " ]]; then
    NUM_CHUNKS=$(echo "$EXTRA_ARGS" | sed -n 's/.*--upload_chunks \([0-9]*\).*/\1/p')
    EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--upload_chunks [0-9]*//g')
fi

# Vast.ai SSH typically uses root and can have changing host keys
SSH_OPTS="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $VAST_PORT"

echo "=================================================="
echo "🚀 Deploying to Vast.ai Instance: $VAST_IP:$VAST_PORT"
echo "Architecture: $ARCH"
echo "Test Mode: $IS_TEST"
echo "=================================================="

echo "[1/4] Creating workspace directories on remote instance..."
ssh $SSH_OPTS root@$VAST_IP "mkdir -p /workspace/Guacarpov"

echo "[2/4] Uploading all code directories and binaries..."
# We explicitly exclude .pt, .npz, and .pth caches/models from the code directories
rsync -az --info=progress2 --exclude="*.pt" --exclude="*.npz" --exclude="*.pth" --exclude="__pycache__" -e "ssh $SSH_OPTS" ./JEPA/ root@$VAST_IP:/workspace/Guacarpov/JEPA/
rsync -az --info=progress2 --exclude="*.pt" --exclude="*.npz" --exclude="*.pth" --exclude="__pycache__" -e "ssh $SSH_OPTS" ./JEPA_Policy/ root@$VAST_IP:/workspace/Guacarpov/JEPA_Policy/
rsync -az --info=progress2 --exclude="*.pt" --exclude="*.npz" --exclude="*.pth" --exclude="__pycache__" -e "ssh $SSH_OPTS" ./JEPA_RL/ root@$VAST_IP:/workspace/Guacarpov/JEPA_RL/
rsync -az --info=progress2 --exclude="__pycache__" -e "ssh $SSH_OPTS" ./stockfish_bin/ root@$VAST_IP:/workspace/Guacarpov/stockfish_bin/
rsync -az --info=progress2 -e "ssh $SSH_OPTS" ./run_pipeline.py root@$VAST_IP:/workspace/Guacarpov/

if [ "$NUM_CHUNKS" == "all" ]; then
    echo "[3/4] Uploading V3 dataset (Streaming all via tar to avoid massive file-overhead)..."
    tar -cf - ./jepa_v3_data/ | ssh $SSH_OPTS root@$VAST_IP "cd /workspace/Guacarpov/ && tar -xf -"
else
    echo "[3/4] Uploading V3 dataset (Streaming $NUM_CHUNKS chunks via tar)..."
    ssh $SSH_OPTS root@$VAST_IP "mkdir -p /workspace/Guacarpov/jepa_v3_data"
    find ./jepa_v3_data/ -maxdepth 1 -type f -name "*.npz" | head -n $NUM_CHUNKS | tar -cf - -T - | ssh $SSH_OPTS root@$VAST_IP "cd /workspace/Guacarpov/ && tar -xf -"
fi

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
    
    echo "Generating launch script..."
    echo "#!/bin/bash" > launch.sh
    echo "set -e" >> launch.sh
    echo "$CMD" >> launch.sh
    chmod +x launch.sh
    
    echo "--------------------------------------------------"
    echo "✅ Setup Complete!"
    echo "The code and dataset are uploaded, and dependencies are installed."
    echo ""
    echo "To start training, SSH into your instance using the following command:"
    echo "  ssh -p $VAST_PORT root@$VAST_IP"
    echo ""
    echo "Then navigate to the workspace and run the launch script:"
    echo "  cd /workspace/Guacarpov"
    echo "  ./launch.sh"
    echo "--------------------------------------------------"
EOF
