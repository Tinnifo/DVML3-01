#!/usr/bin/bash -l

#SBATCH --job-name=INIT_OUR_SWEEP
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tolaso24@student.aau.dk

PYTORCH_CONTAINER=/ceph/container/pytorch/pytorch_25.10.sif
BASEFOLDER=/ceph/home/student.aau.dk/db56hw/revisitingkmers-p3
SCRIPT_PATH=$BASEFOLDER/sweeps/run_our_sweep.py

# Change to base folder
cd $BASEFOLDER

# Run sweep initialization inside container
srun singularity exec $PYTORCH_CONTAINER python $SCRIPT_PATH --init-only

echo ""
echo "=========================================="
echo "Sweep initialization complete!"
echo "Check the output above for the sweep ID."
echo "Use the sweep ID to submit agent jobs with:"
echo "  sbatch sweeps/slurm_run_our_sweep_agent.sh"
echo "=========================================="

