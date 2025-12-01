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
- `--eval_data_dir`: Path to evaluation data directory (enables automatic evaluation)
- `--eval_species`: Comma-separated species list (default: auto-detected)
- `--eval_sample`: Comma-separated sample IDs (default: `5,6`)

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
- `--eval_data_dir`: Path to evaluation data directory (enables automatic evaluation)
- `--eval_species`: Comma-separated species list (default: auto-detected)
- `--eval_sample`: Comma-separated sample IDs (default: `5,6`)

## Running Hyperparameter Sweeps

### Local Sweeps

**Important:** Sweeps are pre-configured to automatically run evaluation after each training run. All metrics are logged to W&B automatically.

1. **Run the sweep** (initializes and runs agent automatically):
   ```bash
   # For OUR model
   python sweeps/run_our_sweep.py
   
   # For REVISIT model
   python sweeps/run_revisit_sweep.py
   ```

   Or initialize separately and run agent manually:
   ```bash
   # Initialize sweep only
   python sweeps/run_our_sweep.py --init-only
   
   # Note the sweep ID from output, then run agent
   wandb agent <entity>/dna-embedding-our/<sweep_id>
   ```

**What happens during a sweep:**
- Each run trains a model with different hyperparameters
- After training, evaluation runs automatically on all species and samples
- All metrics are logged to W&B (e.g., `eval/reference_sample5/f1_at_0.5`)
- The sweep uses `eval/f1_at_0.5` to optimize hyperparameters
- View results in your W&B dashboard automatically

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

### Automatic Evaluation During Training

**Evaluation runs automatically after training** when `--eval_data_dir` is provided. The evaluation will:

- Evaluate on **all available species** (reference, marine, plant) automatically
- Evaluate on **all specified samples** (default: 5, 6)
- Log all metrics to W&B with clear prefixes (e.g., `eval/reference_sample5/f1_at_0.5`)
- Use `eval/f1_at_0.5` as the primary metric for sweep optimization

**Example with automatic evaluation:**

```bash
python src/OUR.py \
    --input train_2m.csv \
    --k 4 \
    --dim 256 \
    --lr 0.001 \
    --epoch 100 \
    --output models/our_model.pt \
    --eval_data_dir . \
    --eval_species reference,marine,plant \
    --eval_sample 5,6 \
    --wandb_project dna-embedding-our
```

**Evaluation parameters:**
- `--eval_data_dir`: Path to evaluation data directory (contains `reference/`, `marine/`, `plant/` subdirectories)
- `--eval_species`: Comma-separated list of species to evaluate (default: `reference,marine,plant`)
  - If not specified, all available species in the data directory will be auto-detected
- `--eval_sample`: Comma-separated list of sample IDs (default: `5,6`)

**Sweep Configuration:**

The sweep configuration files (`sweeps/our_sweep.yaml` and `sweeps/revisit_sweep.yaml`) are pre-configured to:
- Enable automatic evaluation (`eval_data_dir: .`)
- Evaluate all species and samples
- Log all metrics to W&B automatically

When you run a sweep, each training run will automatically:
1. Train the model
2. Evaluate on all species and samples
3. Log comprehensive metrics to W&B
4. Use `eval/f1_at_0.5` for hyperparameter optimization

### Standalone Evaluation

For evaluating pre-trained models without training, use the standalone evaluation script:

```bash
python evaluation/binning.py \
    --test_model_dir models/our_model.pt \
    --model_list our \
    --data_dir . \
    --species reference,marine,plant \
    --samples 5,6 \
    --k 4 \
    --output evaluation_results.txt
```

For evaluating models from W&B sweeps, see `evaluation/evaluate_sweep_models.py` and `evaluation/EVALUATE_SWEEPS.md`.

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
│   ├── binning.py              # Binning evaluation script
│   ├── utils.py                # Evaluation utilities
│   ├── evaluate_sweep_models.py # Script to evaluate models from W&B sweeps
│   └── EVALUATE_SWEEPS.md      # Guide for evaluating sweep models
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
   - The sweep configs default to `cpu` for Mac compatibility
   - CUDA will automatically fall back to CPU if not available

2. **W&B authentication error**: Run `wandb login`

3. **Sweep ID not found**: Check initialization output file or provide sweep ID explicitly
   - The script automatically handles entity mismatches

4. **File not found**: Ensure data paths are correct and files exist
   - Evaluation data should be in the current directory or specify `--eval_data_dir`

5. **Evaluation not running**: 
   - Ensure `--eval_data_dir` is set (defaults to `.` in sweep configs)
   - Check that evaluation data directories (`reference/`, `marine/`, `plant/`) exist
   - Verify that `clustering_0.tsv` and `binning_*.tsv` files exist in each species directory

6. **Device errors**: 
   - If CUDA is requested but not available, the code automatically falls back to CPU
   - Check device availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help

- Check sweep configuration files: `sweeps/*.yaml`
- Review Slurm documentation: `sweeps/README_SLURM.md`
- Check W&B dashboard for experiment logs

## License

[Add your license information here]

