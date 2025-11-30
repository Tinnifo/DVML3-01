# DNA Embedding Models - OUR and REVISIT

This project implements two DNA sequence embedding models using k-mer representations:
- **OUR**: Supervised Contrastive Learning model with multi-view representations
- **REVISIT**: Pairwise learning model with negative sampling

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Models Locally](#running-models-locally)
- [Running Hyperparameter Sweeps](#running-hyperparameter-sweeps)
  - [Local Sweeps](#local-sweeps)
  - [Slurm Cluster Sweeps](#slurm-cluster-sweeps)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd DVML3-01
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases** (for experiment tracking):
   ```bash
   wandb login
   ```

## Quick Start

### Running a Single Training Run

**OUR Model:**
```bash
python src/OUR.py \
    --input train_2m.csv \
    --k 4 \
    --dim 256 \
    --lr 0.001 \
    --epoch 100 \
    --batch_size 64 \
    --output models/our_model.pt \
    --wandb_project dna-embedding-our
```

**REVISIT Model:**
```bash
python src/REVISIT.py \
    --input train_2m.csv \
    --k 4 \
    --dim 256 \
    --lr 0.001 \
    --epoch 100 \
    --batch_size 64 \
    --neg_sample_per_pos 1000 \
    --loss_name bern \
    --output models/revisit_model.pt \
    --wandb_project dna-embedding-revisit
```

## Running Models Locally

### OUR Model

The OUR model uses Supervised Contrastive Learning with multi-view sequence representations.

**Basic usage:**
```bash
python src/OUR.py \
    --input <training_file.csv> \
    --k <kmer_size> \
    --dim <embedding_dimension> \
    --lr <learning_rate> \
    --epoch <num_epochs> \
    --output <model_output_path>
```

**Key parameters:**
- `--input`: Path to training CSV file (format: `left_read,right_read` per line)
- `--k`: k-mer size (typically 2-6)
- `--dim`: Embedding dimension (128, 256, or 512)
- `--lr`: Learning rate
- `--epoch`: Number of training epochs
- `--batch_size`: Batch size (0 = full dataset)
- `--max_views_per_read`: Maximum views per read (2, 4, or 8)
- `--temperature`: Temperature for contrastive loss
- `--wandb_project`: W&B project name (optional)

### REVISIT Model

The REVISIT model uses pairwise learning with negative sampling.

**Basic usage:**
```bash
python src/REVISIT.py \
    --input <training_file.csv> \
    --k <kmer_size> \
    --dim <embedding_dimension> \
    --lr <learning_rate> \
    --epoch <num_epochs> \
    --output <model_output_path>
```

**Key parameters:**
- `--input`: Path to training CSV file (format: `left_read,right_read` per line)
- `--k`: k-mer size (typically 2-6)
- `--dim`: Embedding dimension (128, 256, or 512)
- `--lr`: Learning rate
- `--epoch`: Number of training epochs
- `--neg_sample_per_pos`: Negative samples per positive pair
- `--loss_name`: Loss function (`bern`, `poisson`, or `hinge`)
- `--wandb_project`: W&B project name (optional)

## Running Hyperparameter Sweeps

### Local Sweeps

1. **Initialize a sweep** (one-time setup):
   ```bash
   # For OUR model
   python sweeps/run_our_sweep.py --init-only
   
   # For REVISIT model
   python sweeps/run_revisit_sweep.py --init-only
   ```

2. **Note the sweep ID** from the output (e.g., `abc123def456`)

3. **Run the agent**:
   ```bash
   # For OUR model
   wandb agent <entity>/dna-embedding-our/<sweep_id>
   
   # For REVISIT model
   wandb agent <entity>/dna-embedding-revisit/<sweep_id>
   ```

   Or use the helper scripts:
   ```bash
   # For OUR model (runs agent automatically)
   python sweeps/run_our_sweep.py
   
   # For REVISIT model (runs agent automatically)
   python sweeps/run_revisit_sweep.py
   ```

### Slurm Cluster Sweeps

For running sweeps on a Slurm cluster (e.g., cloud server):

1. **Initialize the sweep**:
   ```bash
   # For OUR model
   sbatch sweeps/slurm_init_our_sweep.sh
   
   # For REVISIT model
   sbatch sweeps/slurm_init_revisit_sweep.sh
   ```

2. **Check the output file** for the sweep ID:
   ```bash
   cat INIT_OUR_SWEEP_<jobid>.out
   # Look for: "Sweep initialized with ID: abc123def456"
   ```

3. **Submit agent jobs** (can submit multiple for parallel execution):
   ```bash
   # Automatic sweep ID detection
   sbatch sweeps/slurm_run_our_sweep_agent.sh
   
   # Or with explicit sweep ID
   sbatch sweeps/slurm_run_our_sweep_agent.sh <SWEEP_ID> [ENTITY] [PROJECT]
   ```

4. **Submit multiple agents** for parallel execution:
   ```bash
   for i in {1..5}; do
       sbatch sweeps/slurm_run_our_sweep_agent.sh <SWEEP_ID>
   done
   ```

**See `sweeps/README_SLURM.md` for detailed Slurm usage instructions.**

## Evaluation

Evaluation can be run automatically after training if `--eval_data_dir` is provided:

```bash
python src/OUR.py \
    --input train_2m.csv \
    --k 4 \
    --dim 256 \
    --lr 0.001 \
    --epoch 100 \
    --output models/our_model.pt \
    --eval_data_dir /path/to/evaluation/data \
    --eval_species reference \
    --eval_sample 5
```

Evaluation metrics (F1 scores, recall at various thresholds) will be logged to W&B automatically.

For standalone evaluation, see `evaluation/binning.py`.

## Project Structure

```
DVML3-01/
├── src/
│   ├── OUR.py              # OUR model (Supervised Contrastive Learning)
│   ├── REVISIT.py          # REVISIT model (Pairwise Learning)
│   └── nonlinear.py        # Shared model utilities
├── sweeps/
│   ├── our_sweep.yaml      # OUR model sweep configuration
│   ├── revisit_sweep.yaml  # REVISIT model sweep configuration
│   ├── run_our_sweep.py    # Local sweep runner for OUR
│   ├── run_revisit_sweep.py # Local sweep runner for REVISIT
│   ├── slurm_init_*.sh     # Slurm initialization scripts
│   ├── slurm_run_*_agent.sh # Slurm agent scripts
│   └── README_SLURM.md     # Detailed Slurm documentation
├── evaluation/
│   ├── binning.py          # Binning evaluation script
│   └── utils.py            # Evaluation utilities
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Data Format

Training data should be a CSV file with one read pair per line:
```
left_read_sequence,right_read_sequence
ATCGATCG...,GCTAGCTA...
...
```

## Key Hyperparameters

### OUR Model
- **k**: k-mer size (2-6)
- **dim**: Embedding dimension (128, 256, 512)
- **max_views_per_read**: Number of views per read (2, 4, 8)
- **temperature**: Contrastive loss temperature (0.05-0.5)

### REVISIT Model
- **k**: k-mer size (2-6)
- **dim**: Embedding dimension (128, 256, 512)
- **neg_sample_per_pos**: Negative samples per positive (100-2000)
- **loss_name**: Loss function (bern, poisson, hinge)

## Monitoring Experiments

All experiments are tracked using Weights & Biases. View your runs at:
- https://wandb.ai/<your-entity>/dna-embedding-our
- https://wandb.ai/<your-entity>/dna-embedding-revisit

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU (`--device cpu`)
2. **W&B authentication error**: Run `wandb login`
3. **Sweep ID not found**: Check initialization output file or provide sweep ID explicitly
4. **File not found**: Ensure data paths are correct and files exist

### Getting Help

- Check sweep configuration files: `sweeps/*.yaml`
- Review Slurm documentation: `sweeps/README_SLURM.md`
- Check W&B dashboard for experiment logs

## License

[Add your license information here]

