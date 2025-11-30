# Slurm Job Scripts for W&B Sweeps

This directory contains Slurm job scripts for running W&B hyperparameter sweeps on the cloud server.

## Overview

The workflow consists of two main steps:
1. **Initialize** a sweep (one-time setup) - creates a sweep ID
2. **Run agents** (can submit multiple jobs) - each agent pulls training runs from the sweep queue

## Files

### Initialization Scripts
- `slurm_init_our_sweep.sh` - Initialize sweep for OUR model
- `slurm_init_revisit_sweep.sh` - Initialize sweep for REVISIT model

### Agent Scripts
- `slurm_run_our_sweep_agent.sh` - Run W&B agent for OUR sweep
- `slurm_run_revisit_sweep_agent.sh` - Run W&B agent for REVISIT sweep

## Usage

### Step 1: Initialize a Sweep

Initialize the sweep (only needs to be done once per sweep):

```bash
# For OUR model
sbatch sweeps/slurm_init_our_sweep.sh

# For REVISIT model
sbatch sweeps/slurm_init_revisit_sweep.sh
```

After the job completes, check the output file (e.g., `INIT_OUR_SWEEP_<jobid>.out`) to find the sweep ID. It will look like:
```
Sweep initialized with ID: abc123def456
```

### Step 2: Submit Agent Jobs

Once you have the sweep ID, submit agent jobs. Each agent job will pull training runs from the sweep queue and execute them.

#### Option A: Automatic Sweep ID Detection
If you submit the agent job from the same directory where the initialization output exists, the script will automatically try to find the sweep ID:

```bash
# For OUR model (will auto-detect sweep ID)
sbatch sweeps/slurm_run_our_sweep_agent.sh

# For REVISIT model (will auto-detect sweep ID)
sbatch sweeps/slurm_run_revisit_sweep_agent.sh
```

#### Option B: Explicit Sweep ID
You can also provide the sweep ID explicitly:

```bash
# For OUR model
sbatch sweeps/slurm_run_our_sweep_agent.sh <SWEEP_ID> [ENTITY] [PROJECT]

# For REVISIT model
sbatch sweeps/slurm_run_revisit_sweep_agent.sh <SWEEP_ID> [ENTITY] [PROJECT]
```

**Examples:**
```bash
# With sweep ID only (uses default project)
sbatch sweeps/slurm_run_our_sweep_agent.sh abc123def456

# With sweep ID and entity
sbatch sweeps/slurm_run_our_sweep_agent.sh abc123def456 myentity

# With sweep ID, entity, and project
sbatch sweeps/slurm_run_our_sweep_agent.sh abc123def456 myentity myproject
```

### Running Multiple Agents in Parallel

You can submit multiple agent jobs to run training runs in parallel. Each agent will pull jobs from the same sweep queue:

```bash
# Submit 5 parallel agents for OUR sweep
for i in {1..5}; do
    sbatch sweeps/slurm_run_our_sweep_agent.sh abc123def456
done
```

## Resource Requirements

### Initialization Scripts
- **CPUs**: 1
- **Memory**: 4G
- **Time**: 10 minutes
- **GPU**: Not required

### Agent Scripts
- **CPUs**: 4
- **Memory**: 24G (maximum available for GPU jobs)
- **Time**: Up to 12 hours (maximum available)
- **GPU**: 1 GPU required

## Configuration

The scripts use the following paths (configured in the scripts):
- **Container**: `/ceph/container/pytorch/pytorch_25.10.sif`
- **Base Folder**: `/ceph/home/student.aau.dk/db56hw/revisitingkmers-p3`
- **Data Directory**: `/ceph/project/p3-kmer/dataset`
- **Email**: `tolaso24@student.aau.dk`

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### View Job Output
```bash
# View initialization output
cat INIT_OUR_SWEEP_<jobid>.out

# View agent output
cat OUR_SWEEP_AGENT_<jobid>.out
```

### Monitor Sweep Progress
View your sweep progress on W&B:
- OUR model: https://wandb.ai/<entity>/dna-embedding-our/sweeps/<sweep_id>
- REVISIT model: https://wandb.ai/<entity>/dna-embedding-revisit/sweeps/<sweep_id>

## Troubleshooting

### Sweep ID Not Found
If the agent script cannot find the sweep ID:
1. Check that the initialization job completed successfully
2. Verify the output file exists: `INIT_OUR_SWEEP_<jobid>.out` or `INIT_REVISIT_SWEEP_<jobid>.out`
3. Manually provide the sweep ID as a command-line argument

### Agent Not Starting
- Ensure W&B is properly configured (API key set)
- Check that the sweep ID is correct
- Verify the project and entity names match your W&B setup

### GPU Issues
- Ensure `--nv` flag is used in singularity exec (already included in scripts)
- Check GPU availability: `squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"`

## Notes

- Each agent job will run until it completes its allocated time (up to 12 hours) or until the sweep queue is empty
- Multiple agents can run simultaneously, each pulling different training runs from the queue
- Evaluation is integrated into the training scripts and will run automatically after each training run completes (if `eval_data_dir` is configured in the sweep YAML)

