# DNA Embedding Models - OUR and REVISIT

Two DNA sequence embedding models using k-mer representations:
- **OUR**: Supervised Contrastive Learning with multi-view representations
- **REVISIT**: Pairwise learning with negative sampling

## Local Setup

### Prerequisites

1. **Python 3.8+**: Ensure Python 3.8 or higher is installed on your system
2. **CUDA (optional)**: For GPU support, install CUDA-compatible PyTorch. The requirements will install CPU version by default.
3. **gdown** (if you don't already have it) for downloading the datasets:
   ```bash
   pip install gdown
   ```

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tinnifo/DVML3-01.git
   cd DVML3-01
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **For GPU support**, install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. **Download datasets**:

   **Training dataset:**
   ```bash
   gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
   unzip dnabert-s_train.zip
   ```

   **Evaluation datasets:**
   ```bash
   gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
   unzip dnabert-s_eval.zip
   ```

5. **Set up Weights & Biases** (if using W&B for tracking):
   ```bash
   wandb login
   ```

6. **Configure W&B entity** (if using team workspace):
   - Edit `WANDB_ENTITY` in `sweeps/run_our_sweep.py` and `sweeps/run_revisit_sweep.py`
   - Set to your W&B entity/team name (e.g., `"tinnifo-aalborg-universitet"`)

**Note:** 
- Make sure your virtual environment is activated before running commands
- The YAML files use relative paths (`train_2m.csv`, `val_48k.csv`) which work correctly with this setup
- All data files, models, and outputs will be created in the project directory
- See [POST_HOC_OPTIMIZATION.md](POST_HOC_OPTIMIZATION.md) for the complete workflow

## Workflow: Post-Hoc Optimization

This project uses a **post-hoc optimization workflow** where training and evaluation are separated:

1. **Train models** with hyperparameter sweeps
2. **Evaluate all models** separately
3. **Analyze results** in W&B to find best hyperparameters
4. **Refine** and repeat

### Step 1: Training Sweeps

Run hyperparameter sweeps to train multiple models:

**For OUR model:**
```bash
python sweeps/run_our_sweep.py --count 50  # Train 50 models
```

**For REVISIT model:**
```bash
python sweeps/run_revisit_sweep.py --count 50
```

**What happens:**
- Each run trains with different hyperparameters
- Models are saved to `models/` directory
- Training hyperparameters logged to W&B config
- Model paths logged to W&B summary
- Sweep optimizes based on `val_loss` (to prevent overfitting)
- **Evaluation is disabled during training** (done separately)

**Configuring sweeps:**
- Edit `sweeps/our_sweep.yaml` to configure the OUR model sweep parameters
- Edit `sweeps/revisit_sweep.yaml` to configure the REVISIT model sweep parameters
- Update `WANDB_ENTITY` in `sweeps/run_our_sweep.py` and `sweeps/run_revisit_sweep.py` to your W&B entity/team

### Step 2: Evaluate All Models

After training, evaluate all models:

```bash
python evaluate_all_models.py \
    --model_dir models/ \
    --model_type our \
    --eval_data_dir . \
    --wandb_project dna-embedding-our \
    --wandb_entity your-entity
```

**What happens:**
- Finds all `.pt` model files in the directory
- Evaluates each model on evaluation datasets
- Links evaluation results to original training runs in W&B
- Logs evaluation metrics (`eval/f1_at_0.5`, `eval/recall_at_0.5`, etc.) to W&B

### Step 3: Analyze in W&B

1. Go to your W&B project: `https://wandb.ai/<entity>/<project>`
2. Use **Parallel Coordinates** or **Hyperparameter Importance** to see which training hyperparameters led to best `eval/f1_at_0.5`
3. Identify promising hyperparameter ranges

### Step 4: Refine and Repeat

Update your sweep configuration based on findings and run another focused sweep.

**For detailed workflow instructions, see [POST_HOC_OPTIMIZATION.md](POST_HOC_OPTIMIZATION.md)**

## Project Structure

```
DVML3-01/
├── src/
│   ├── OUR.py              # OUR model training
│   └── REVISIT.py          # REVISIT model training
├── sweeps/
│   ├── our_sweep.yaml      # OUR sweep configuration
│   ├── revisit_sweep.yaml  # REVISIT sweep configuration
│   ├── evaluation_sweep.yaml  # Evaluation parameter sweep (optional)
│   ├── run_our_sweep.py    # OUR sweep runner
│   └── run_revisit_sweep.py # REVISIT sweep runner
├── evaluation/
│   └── utils.py            # Evaluation utilities
├── evaluate_models.py      # Single model evaluation script
├── evaluate_all_models.py  # Batch evaluation script (main workflow)
├── train_2m.csv            # Training data
├── val_48k.csv             # Validation data
├── reference/              # Evaluation data
├── marine/                 # Evaluation data
├── plant/                   # Evaluation data
├── models/                  # Model outputs (created automatically)
├── POST_HOC_OPTIMIZATION.md # Detailed workflow guide
├── FILE_USAGE.md           # File usage documentation
├── requirements.txt        # Python dependencies
└── README.md
```

## Data Format

Training data should be a CSV file with one read pair per line:
```
left_read_sequence,right_read_sequence
ATCGATCG...,GCTAGCTA...
...
```

## Evaluation

### Single Model Evaluation

Evaluate a single model:
```bash
python evaluate_models.py \
    --model_path models/our_model.pt \
    --model_type our \
    --eval_data_dir .
```

### Batch Evaluation (Recommended)

Evaluate all models and link to training runs:
```bash
python evaluate_all_models.py \
    --model_dir models/ \
    --model_type our \
    --eval_data_dir . \
    --wandb_project dna-embedding-our \
    --wandb_entity your-entity
```

**Note:** Evaluation runs on CPU automatically (no GPU needed).

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size in sweep YAML config
2. **W&B authentication error**: Run `wandb login` in your terminal
3. **W&B entity/permission error**: Update `WANDB_ENTITY` in sweep runner scripts
4. **File not found**: Check data paths in YAML configuration files and ensure data files exist
5. **Evaluation not linking to training runs**: Ensure `--wandb_project` matches your training project
6. **Device errors**: Code automatically falls back to CPU if CUDA unavailable
7. **Import errors**: Make sure you're in the project root directory and your virtual environment is activated
8. **Model not found**: Check that models are saved in the `models/` directory after training

## Monitoring and Analysis

View experiments in Weights & Biases:
- Training runs: https://wandb.ai/<your-entity>/dna-embedding-our
- Evaluation runs: Tagged with `post_hoc_evaluation` in the same project

**Key W&B features for analysis:**
- **Parallel Coordinates**: See which hyperparameters lead to best `eval/f1_at_0.5`
- **Hyperparameter Importance**: Identify which training parameters matter most
- **Compare Runs**: Compare training configs side-by-side with evaluation results

## Additional Documentation

- **[POST_HOC_OPTIMIZATION.md](POST_HOC_OPTIMIZATION.md)**: Detailed workflow guide for post-hoc optimization
- **[FILE_USAGE.md](FILE_USAGE.md)**: Documentation of which files are used in which workflows
