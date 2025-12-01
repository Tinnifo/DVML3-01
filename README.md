# DNA Embedding Models - OUR and REVISIT

Two DNA sequence embedding models using k-mer representations:
- **OUR**: Supervised Contrastive Learning with multi-view representations
- **REVISIT**: Pairwise learning with negative sampling

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tinnifo/DVML3-01.git
   cd DVML3-01
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases** (optional, for experiment tracking):
   ```bash
   wandb login
   ```

4. **Install gdown** (if you don't already have it) for downloading the datasets:
   ```bash
   pip install gdown
   ```

## Datasets

To download and prepare the training dataset, run the following commands:

```bash
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
unzip dnabert-s_train.zip
```

To download the evaluation datasets, use the following commands:

```bash
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
unzip dnabert-s_eval.zip
```

## Docker Setup (Recommended)

Docker ensures the project works on all computers regardless of local Python/package setup.

### Prerequisites

1. **Install Docker**: Download and install Docker from [docker.com](https://www.docker.com/get-started)
2. **Install Docker Compose** (optional, for easier management)
3. **For GPU support**: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Setup Steps

1. **Download datasets** (see [Datasets](#datasets) section above) to your local machine

2. **Build the Docker image**:
   ```bash
   docker build -t dna-embedding .
   ```

3. **Prepare your data directory structure**:
   ```
   your-project/
   ├── data/
   │   ├── train_2m.csv          # Training data
   │   ├── reference/            # Evaluation data
   │   ├── marine/               # Evaluation data
   │   └── plant/                 # Evaluation data
   ├── models/                    # Model outputs (created automatically)
   └── DVML3-01/                 # This repository
   ```

4. **Run sweeps with Docker**:

   **For GPU (recommended):**
   ```bash
   # For OUR model
   docker run --gpus all \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/wandb:/app/wandb \
     dna-embedding python sweeps/run_our_sweep.py

   # For REVISIT model
   docker run --gpus all \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/wandb:/app/wandb \
     dna-embedding python sweeps/run_revisit_sweep.py
   ```

   **For CPU-only:**
   ```bash
   docker run \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/wandb:/app/wandb \
     dna-embedding python sweeps/run_our_sweep.py
   ```

5. **Update YAML configuration files** to use Docker paths:
   - In `sweeps/our_sweep.yaml` and `sweeps/revisit_sweep.yaml`, set:
     - `input: /app/data/train_2m.csv`
     - `eval_data_dir: /app/data`
     - `output: /app/models/our_model.pt` (or `revisit_model.pt`)

**Note:** 
- Data files are mounted as volumes, so they remain on your host machine
- Model outputs and W&B logs are saved to your host machine via volume mounts
- The Docker container provides a consistent environment across all systems

Docker ensures the project works on all computers regardless of local Python/package setup.

1. **Build the Docker image**:
   ```bash
   docker build -t dna-embedding .
   ```

2. **Download your data** to your local machine (e.g., `./data/` directory)

3. **Run sweeps with Docker** (mount your data directory):
   ```bash
   # For OUR model (with GPU)
   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dna-embedding python sweeps/run_our_sweep.py

   # For REVISIT model (with GPU)
   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dna-embedding python sweeps/run_revisit_sweep.py

   # For CPU-only (no GPU)
   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dna-embedding python sweeps/run_our_sweep.py
   ```

**Note:** Make sure your YAML configuration files point to the correct data paths (e.g., `/app/data/train_2m.csv` and `/app/data/` for evaluation data).

## Running Hyperparameter Sweeps

Sweeps automatically run evaluation after each training run and log all metrics to W&B.

**Run a sweep:**
```bash
# For OUR model
python sweeps/run_our_sweep.py

# For REVISIT model
python sweeps/run_revisit_sweep.py
```

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
│   ├── utils.py            # Evaluation utilities (required)
│   └── binning.py          # Standalone evaluation script (optional)
├── requirements.txt
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
2. **W&B authentication error**: Run `wandb login`
3. **File not found**: Check data paths in YAML configuration files
4. **Evaluation not running**: Ensure `eval_data_dir` is set in YAML and evaluation data directories exist
5. **Device errors**: Code automatically falls back to CPU if CUDA unavailable

## Monitoring

View experiments in Weights & Biases:
- https://wandb.ai/<your-entity>/dna-embedding-our
- https://wandb.ai/<your-entity>/dna-embedding-revisit
