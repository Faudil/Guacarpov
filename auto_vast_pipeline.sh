#!/bin/bash

# auto_vast_pipeline.sh
# End-to-end automation for reserving, deploying, and tearing down Vast.ai instances.

if [ "$#" -lt 1 ]; then
    echo "=========================================================================="
    echo "Usage:"
    echo "  ./auto_vast_pipeline.sh full <OFFER_ID> [ARCH] [FLAGS...]   (Deploy, monitor, and destroy automatically)"
    echo "  ./auto_vast_pipeline.sh start <OFFER_ID> [ARCH] [FLAGS...]  (Deploy and exit. Safe to turn off PC!)"
    echo "  ./auto_vast_pipeline.sh recover <INSTANCE_ID>               (Download weights and destroy instance)"
    echo "=========================================================================="
    exit 1
fi

COMMAND=$1
shift

if ! command -v vastai &> /dev/null || ! command -v jq &> /dev/null; then
    echo "❌ vastai CLI or jq not found. Please install them."
    exit 1
fi

get_instance_ssh() {
    local ID=$1
    local STATE_OUT=$(vastai show instances --raw 2>/dev/null | jq -r ".[] | select(type == \"object\" and .id == $ID)")
    SSH_HOST=$(echo "$STATE_OUT" | jq -r '.ssh_host')
    SSH_PORT=$(echo "$STATE_OUT" | jq -r '.ssh_port')
    ACTUAL_STATUS=$(echo "$STATE_OUT" | jq -r '.actual_status')
}

do_start() {
    local OFFER_ID=$1
    shift
    local ARCH="spatiotemporal"
    if [[ $# -gt 0 && ! "$1" == --* ]]; then
        ARCH=$1
        shift
    fi
    local EXTRA_ARGS="$@"

    echo "=================================================="
    echo "🚀 Reserving Vast.ai Instance from Offer: $OFFER_ID"
    echo "Architecture: $ARCH"
    echo "=================================================="

    local CREATE_OUT=$(vastai create instance $OFFER_ID --image pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel --disk 50 --raw)
    INSTANCE_ID=$(echo "$CREATE_OUT" | jq '.new_contract')

    if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "null" ]; then
        echo "❌ Failed to create instance."
        exit 1
    fi

    echo "✅ Instance $INSTANCE_ID created successfully."
    echo "⏳ Waiting for instance to boot and SSH to become available..."

    trap 'echo "User aborted. Tearing down instance..."; vastai destroy instance $INSTANCE_ID; exit 1' SIGINT SIGTERM

    while true; do
        get_instance_ssh $INSTANCE_ID
        if [ "$ACTUAL_STATUS" == "running" ] && [ "$SSH_HOST" != "null" ] && [ "$SSH_PORT" != "null" ]; then
            echo "✅ Instance is running! IP: $SSH_HOST, Port: $SSH_PORT"
            break
        elif [[ "$ACTUAL_STATUS" == *"error"* || "$ACTUAL_STATUS" == "offline" ]]; then
            echo "❌ Instance failed to boot. Destroying instance."
            vastai destroy instance $INSTANCE_ID
            exit 1
        fi
        sleep 10
    done

    echo "⏳ Waiting 20 seconds for SSH daemon to fully initialize..."
    sleep 20

    set +e
    ./vast_deploy.sh $SSH_HOST $SSH_PORT $ARCH $EXTRA_ARGS
    local DEPLOY_STATUS=$?
    if [ $DEPLOY_STATUS -ne 0 ]; then
        echo "⚠️ vast_deploy.sh encountered an error (Code $DEPLOY_STATUS)."
    else
        echo "✅ Deployment successful. Training is now running in tmux session 'jepa_train'."
    fi
    trap - SIGINT SIGTERM
}

do_monitor() {
    echo "=================================================="
    echo "👀 Monitoring Training Progress (Every 2 mins)"
    echo "=================================================="
    local SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $SSH_PORT"
    mkdir -p recovered_weights

    while true; do
        echo "[$(date +'%H:%M:%S')] Syncing artifacts..."
        rsync -azq -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/JEPA/chess_jepa.pth ./recovered_weights/ || true
        rsync -azq -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/JEPA_Policy/chess_sft_policy.pth ./recovered_weights/ || true
        rsync -azq -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/JEPA_RL/chess_rl_policy.pth ./recovered_weights/ || true
        rsync -azq -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/training.log ./recovered_weights/ || true

        ssh $SSH_OPTS root@$SSH_HOST "tmux has-session -t jepa_train 2>/dev/null"
        local TMUX_STATUS=$?
        
        if [ $TMUX_STATUS -eq 1 ]; then
            echo "✅ Tmux session 'jepa_train' has ended (Training finished or crashed)."
            break
        elif [ $TMUX_STATUS -eq 255 ]; then
            echo "⚠️ SSH Connection dropped (Broken Pipe). Retrying..."
        elif [ $TMUX_STATUS -ne 0 ]; then
            echo "⚠️ Unknown error ($TMUX_STATUS). Retrying..."
        fi
        sleep 120
    done
}

do_recover() {
    local INSTANCE_ID=$1
    get_instance_ssh $INSTANCE_ID
    
    if [ "$SSH_HOST" == "null" ]; then
        echo "❌ Instance $INSTANCE_ID not found or not running."
        exit 1
    fi

    echo "=================================================="
    echo "📥 Final Recovery of Artifacts and Weights for $INSTANCE_ID"
    echo "=================================================="
    local SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $SSH_PORT"
    mkdir -p recovered_weights

    rsync -avzP -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/JEPA/chess_jepa.pth ./recovered_weights/ || true
    rsync -avzP -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/JEPA_Policy/chess_sft_policy.pth ./recovered_weights/ || true
    rsync -avzP -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/JEPA_RL/chess_rl_policy.pth ./recovered_weights/ || true
    rsync -avzP -e "ssh $SSH_OPTS" root@$SSH_HOST:/workspace/Guacarpov/training.log ./recovered_weights/ || true

    echo "=================================================="
    echo "🔥 Tearing Down Instance"
    echo "=================================================="
    vastai destroy instance $INSTANCE_ID
    echo "✅ Instance destroyed. Pipeline complete! Check 'recovered_weights'."
}

if [ "$COMMAND" == "full" ]; then
    do_start "$@"
    do_monitor
    do_recover $INSTANCE_ID
elif [ "$COMMAND" == "start" ]; then
    do_start "$@"
    echo "=================================================="
    echo "🎉 Start Phase Complete!"
    echo "You can now safely turn off your PC."
    echo "When you want to retrieve your weights and destroy the instance, run:"
    echo "./auto_vast_pipeline.sh recover $INSTANCE_ID"
    echo "=================================================="
elif [ "$COMMAND" == "monitor" ]; then
    if [ -z "$1" ]; then
        echo "❌ Missing INSTANCE_ID. Usage: ./auto_vast_pipeline.sh monitor <INSTANCE_ID>"
        exit 1
    fi
    INSTANCE_ID=$1
    get_instance_ssh $INSTANCE_ID
    if [ "$SSH_HOST" == "null" ]; then
        echo "❌ Instance $INSTANCE_ID not found or not running."
        exit 1
    fi
    do_monitor
elif [ "$COMMAND" == "redeploy" ]; then
    if [ -z "$1" ]; then
        echo "❌ Missing INSTANCE_ID. Usage: ./auto_vast_pipeline.sh redeploy <INSTANCE_ID> [ARCH] [FLAGS...]"
        exit 1
    fi
    INSTANCE_ID=$1
    shift
    ARCH="spatiotemporal"
    if [[ $# -gt 0 && ! "$1" == --* ]]; then
        ARCH=$1
        shift
    fi
    EXTRA_ARGS="$@"
    
    get_instance_ssh $INSTANCE_ID
    if [ -z "$SSH_HOST" ] || [ "$SSH_HOST" == "null" ]; then
        echo "❌ Instance $INSTANCE_ID not found, not running, or failed to parse IP."
        exit 1
    fi
    
    echo "=================================================="
    echo "🔄 Redeploying to existing instance $INSTANCE_ID"
    echo "IP: $SSH_HOST, Port: $SSH_PORT"
    echo "=================================================="
    ./vast_deploy.sh "$SSH_HOST" "$SSH_PORT" "$ARCH" $EXTRA_ARGS
elif [ "$COMMAND" == "recover" ]; then
    if [ -z "$1" ]; then
        echo "❌ Missing INSTANCE_ID. Usage: ./auto_vast_pipeline.sh recover <INSTANCE_ID>"
        exit 1
    fi
    do_recover $1
else
    echo "❌ Unknown command: $COMMAND. Use 'full', 'start', 'redeploy', 'monitor', or 'recover'."
    exit 1
fi
