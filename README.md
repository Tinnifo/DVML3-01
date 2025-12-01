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

6. **Run sweeps**:

   **For OUR model:**
   ```bash
   python sweeps/run_our_sweep.py
   ```

   **For REVISIT model:**
   ```bash
   python sweeps/run_revisit_sweep.py
   ```

**Note:** 
- Make sure your virtual environment is activated before running commands
- The YAML files use relative paths (`train_2m.csv`, `.` for eval_data_dir) which work correctly with this setup
- All data files, models, and outputs will be created in the project directory

## Running Hyperparameter Sweeps

Sweeps automatically run evaluation after each training run and log all metrics to W&B.

**What happens:**
- Each run trains with different hyperparameters
- Evaluation runs automatically after training
- All metrics logged to W&B (e.g., `eval/f1_at_0.5`, `eval/reference_sample5/f1_at_0.5`)
- Sweep optimizes based on `eval/f1_at_0.5` (aggregated across all species/samples)

**Configuring sweeps:**
- Edit `sweeps/our_sweep.yaml` to configure the OUR model sweep parameters
- Edit `sweeps/revisit_sweep.yaml` to configure the REVISIT model sweep parameters
- You can change hyperparameters, evaluation settings, data paths, and W&B project settings in these YAML files

## Project Structure

```
DVML3-01/
├── src/
│   ├── OUR.py              # OUR model
│   └── REVISIT.py          # REVISIT model
├── sweeps/
│   ├── our_sweep.yaml      # OUR sweep configuration
│   ├── revisit_sweep.yaml # REVISIT sweep configuration
│   ├── run_our_sweep.py    # OUR sweep runner
│   └── run_revisit_sweep.py # REVISIT sweep runner
├── evaluation/
│   └── utils.py            # Evaluation utilities (required)
├── train_2m.csv            # Training data
├── reference/              # Evaluation data
├── marine/                 # Evaluation data
├── plant/                   # Evaluation data
├── models/                  # Model outputs (created automatically)
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

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or use `--device cpu` in YAML config
2. **W&B authentication error**: Run `wandb login` in your terminal
3. **File not found**: Check data paths in YAML configuration files and ensure data files exist in the project directory
4. **Evaluation not running**: Ensure `eval_data_dir` is set in YAML and evaluation data directories exist
5. **Device errors**: Code automatically falls back to CPU if CUDA unavailable
6. **Import errors**: Make sure you're in the project root directory and your virtual environment is activated
7. **Python path issues**: If you encounter module import errors, ensure you're running commands from the project root directory

## Monitoring

View experiments in Weights & Biases:
- https://wandb.ai/<your-entity>/dna-embedding-our
- https://wandb.ai/<your-entity>/dna-embedding-revisit
