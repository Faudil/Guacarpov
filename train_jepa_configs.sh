#!/bin/bash

# Default values
SIZE=${1:-"medium"}
ARCH=${2:-"spatiotemporal"}
BATCH_SIZE=${3:-""}
NUM_GAMES=${4:-""}

# Help message
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./train_jepa_configs.sh [small|medium|big] [spatiotemporal|resnet|convnext|vit] [batch_size] [num_games]"
    echo "Default size: medium"
    echo "Default arch (jepa_type): spatiotemporal"
    echo "Default batch_size: 256 (512 for small ResNet/ConvNeXt/ViT)"
    echo "Default num_games: 50,000 (small), 500,000 (medium), 3,000,000 (big)"
    exit 0
fi

# Validate size
if [[ "$SIZE" != "small" && "$SIZE" != "medium" && "$SIZE" != "big" ]]; then
    echo "Error: Invalid size '$SIZE'. Must be one of: small, medium, big"
    exit 1
fi

# Validate arch
if [[ "$ARCH" != "spatiotemporal" && "$ARCH" != "resnet" && "$ARCH" != "convnext" && "$ARCH" != "vit" ]]; then
    echo "Error: Invalid architecture '$ARCH'. Must be one of: spatiotemporal, resnet, convnext, vit"
    exit 1
fi

# Validate batch size if provided
if [ -n "$BATCH_SIZE" ]; then
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -le 0 ]; then
        echo "Error: Invalid batch size '$BATCH_SIZE'. Must be a positive integer."
        exit 1
    fi
    B_SIZE="$BATCH_SIZE"
else
    # Default batch sizes
    if [ "$SIZE" == "small" ] && [ "$ARCH" != "spatiotemporal" ]; then
        B_SIZE="512"
    else
        B_SIZE="256"
    fi
fi

# Validate and set number of games
if [ -n "$NUM_GAMES" ]; then
    if ! [[ "$NUM_GAMES" =~ ^[0-9]+$ ]] || [ "$NUM_GAMES" -le 0 ]; then
        echo "Error: Invalid number of games '$NUM_GAMES'. Must be a positive integer."
        exit 1
    fi
    N_GAMES="$NUM_GAMES"
else
    # Default game limits based on model size to prevent small models from running forever on 30M games
    if [ "$SIZE" == "small" ]; then
        N_GAMES="50000"
    elif [ "$SIZE" == "medium" ]; then
        N_GAMES="500000"
    else
        N_GAMES="3000000"
    fi
fi

echo "=================================================="
echo "Preparing training run:"
echo "  • Config Size  : $SIZE"
echo "  • JEPA Type    : $ARCH"
echo "  • Batch Size   : $B_SIZE"
echo "  • Num Games    : $N_GAMES"
echo "=================================================="

# Base command
CMD="venv/bin/python JEPA/train.py --arch $ARCH --data_dir jepa_v3_data --in_channels 111 --epochs 5 --num_workers 4 --batch_size $B_SIZE --num_games $N_GAMES"

# Add config specific options
if [ "$ARCH" == "spatiotemporal" ]; then
    if [ "$SIZE" == "small" ]; then
        CMD="$CMD --latent_dim 256 --spatial_dim 64 --spatial_blocks 4 --temporal_layers 4 --temporal_heads 8 --predictor_layers 6 --encoder_lr 5e-4 --predictor_lr 1e-3 --encoder_weight_decay 1e-4 --predictor_weight_decay 0.0"
    elif [ "$SIZE" == "medium" ]; then
        CMD="$CMD --latent_dim 384 --spatial_dim 96 --spatial_blocks 8 --temporal_layers 6 --temporal_heads 12 --predictor_layers 8 --encoder_lr 5e-4 --predictor_lr 1e-3 --encoder_weight_decay 1e-4 --predictor_weight_decay 0.0"
    elif [ "$SIZE" == "big" ]; then
        CMD="$CMD --latent_dim 512 --spatial_dim 128 --spatial_blocks 12 --temporal_layers 6 --temporal_heads 16 --predictor_layers 10 --encoder_lr 5e-4 --predictor_lr 1e-3 --encoder_weight_decay 1e-4 --predictor_weight_decay 0.0"
    fi
else
    # ResNet, ConvNeXt, ViT configs
    if [ "$SIZE" == "small" ]; then
        CMD="$CMD --latent_dim 256 --num_res_blocks 6 --num_filters 64 --lr 1e-3 --weight_decay 1e-4"
    elif [ "$SIZE" == "medium" ]; then
        CMD="$CMD --latent_dim 384 --num_res_blocks 10 --num_filters 128 --lr 1e-3 --weight_decay 1e-4"
    elif [ "$SIZE" == "big" ]; then
        CMD="$CMD --latent_dim 512 --num_res_blocks 16 --num_filters 256 --lr 1e-3 --weight_decay 1e-4"
    fi
fi

# Run the command
echo "Running command:"
echo "  $CMD"
echo "=================================================="
$CMD