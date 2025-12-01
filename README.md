# DNA Embedding Models - OUR and REVISIT

Two DNA sequence embedding models using k-mer representations:
- **OUR**: Supervised Contrastive Learning with multi-view representations
- **REVISIT**: Pairwise learning with negative sampling

## Docker Setup

Docker ensures the project works on all computers regardless of local Python/package setup.

### Prerequisites

1. **Install Docker**: Download and install Docker from [docker.com](https://www.docker.com/get-started)
2. **For GPU support**: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. **Install gdown** (if you don't already have it) for downloading the datasets:
   ```bash
   pip install gdown
   ```

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tinnifo/DVML3-01.git
   cd DVML3-01
   ```

2. **Download datasets** to your local machine:

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

3. **Build the Docker image**:
   ```bash
   docker build -t dna-embedding .
   ```

4. **Run sweeps with Docker**:

   **For GPU (recommended):**
   ```bash
   # For OUR model
   docker run --gpus all \
     -v $(pwd):/app \
     -w /app \
     dna-embedding python sweeps/run_our_sweep.py

   # For REVISIT model
   docker run --gpus all \
     -v $(pwd):/app \
     -w /app \
     dna-embedding python sweeps/run_revisit_sweep.py
   ```

   **For CPU-only:**
   ```bash
   docker run \
     -v $(pwd):/app \
     -w /app \
     dna-embedding python sweeps/run_our_sweep.py
   ```

**Note:** 
- The entire project directory is mounted to `/app` in the container
- All data files, models, and outputs remain on your host machine
- The YAML files use relative paths (`train_2m.csv`, `.` for eval_data_dir) which work correctly with this setup
- The Docker container provides a consistent environment across all systems

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
├── Dockerfile              # Docker configuration
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
2. **W&B authentication error**: Run `wandb login` inside the Docker container or mount your W&B credentials
3. **File not found**: Check data paths in YAML configuration files and ensure data is mounted correctly
4. **Evaluation not running**: Ensure `eval_data_dir` is set in YAML and evaluation data directories exist
5. **Device errors**: Code automatically falls back to CPU if CUDA unavailable

## Monitoring

View experiments in Weights & Biases:
- https://wandb.ai/<your-entity>/dna-embedding-our
- https://wandb.ai/<your-entity>/dna-embedding-revisit
