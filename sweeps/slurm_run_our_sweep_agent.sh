#!/usr/bin/bash -l

#SBATCH --job-name=OUR_SWEEP_AGENT
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tolaso24@student.aau.dk

# Usage: sbatch sweeps/slurm_run_our_sweep_agent.sh [SWEEP_ID] [ENTITY] [PROJECT]
# Example: sbatch sweeps/slurm_run_our_sweep_agent.sh abc123def myentity myproject
# If SWEEP_ID is not provided, it will be read from the most recent initialization output

PYTORCH_CONTAINER=/ceph/container/pytorch/pytorch_25.10.sif
BASEFOLDER=/ceph/home/student.aau.dk/db56hw/revisitingkmers-p3
DATA_DIR=/ceph/project/p3-kmer/dataset

# Default values (can be overridden by command line arguments)
SWEEP_ID=${1:-""}
ENTITY=${2:-""}
PROJECT=${3:-"dna-embedding-our"}

# Change to base folder
cd $BASEFOLDER

# If sweep ID not provided, try to extract from latest init output
if [ -z "$SWEEP_ID" ]; then
    echo "Warning: SWEEP_ID not provided. Attempting to find from recent initialization..."
    # Look for the most recent initialization output file (check current dir and sweeps dir)
    LATEST_INIT=$(ls -t INIT_OUR_SWEEP_*.out ${BASEFOLDER}/sweeps/INIT_OUR_SWEEP_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_INIT" ]; then
        # Extract sweep ID using sed (more portable than grep -oP)
        SWEEP_ID=$(grep "Sweep initialized with ID:" "$LATEST_INIT" 2>/dev/null | sed -n 's/.*Sweep initialized with ID: \([^[:space:]]*\).*/\1/p' | head -1)
        if [ -n "$SWEEP_ID" ]; then
            echo "Found sweep ID from $LATEST_INIT: $SWEEP_ID"
        fi
    fi
fi

# Validate sweep ID
if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID is required!"
    echo "Usage: sbatch sweeps/slurm_run_our_sweep_agent.sh [SWEEP_ID] [ENTITY] [PROJECT]"
    echo "Or ensure INIT_OUR_SWEEP job has completed and output file exists."
    exit 1
fi

# Build wandb agent command
if [ -n "$ENTITY" ]; then
    WANDB_AGENT_CMD="wandb agent ${ENTITY}/${PROJECT}/${SWEEP_ID}"
else
    WANDB_AGENT_CMD="wandb agent ${PROJECT}/${SWEEP_ID}"
fi

echo "=========================================="
echo "Starting W&B sweep agent"
echo "Sweep ID: $SWEEP_ID"
echo "Project: $PROJECT"
echo "Entity: ${ENTITY:-<default>}"
echo "Command: $WANDB_AGENT_CMD"
echo "=========================================="

# Run W&B agent inside container
srun singularity exec --nv $PYTORCH_CONTAINER bash -c "cd $BASEFOLDER && $WANDB_AGENT_CMD"

echo ""
echo "=========================================="
echo "Agent job completed!"
echo "=========================================="

