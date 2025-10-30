#!/bin/bash

# Network configuration
export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1

# SLURM variables
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base paths
BASE_DIR="path/to/base"
SCRIPT_PATH="$BASE_DIR/main.py"

# Configuration file (passed as first argument)
CONFIG_FILE=${1:-"$BASE_DIR/slurm-scripts/config-gauss.json"}

echo "=============================================================="
echo "Training Launch"
echo "=============================================================="
echo "Node: $SLURMD_NODENAME"
echo "Process: $SLURM_PROCID/$SLURM_JOB_NUM_NODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Config: $CONFIG_FILE"
echo "=============================================================="

# run training
python -u "$SCRIPT_PATH" --config_file="$CONFIG_FILE"

echo "Training completed with exit code: $?"