#!/bin/bash
#SBATCH --job-name=gauss
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=../outputs/%j-%x.out
#SBATCH --error=../outputs/%j-%x.err
#SBATCH --qos=normal

source /path/to/env/bin/activate
cd /path/to/slurm-scripts

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

echo "World Size: $WORLD_SIZE"
nvidia-smi

CONFIG_FILE="./config-gauss.json"

echo "START $SLURM_JOBID: $(date)"

# Launch distributed training (one process per GPU)
srun --label launch-train.sh "$CONFIG_FILE"

echo "END $SLURM_JOBID: $(date)"
