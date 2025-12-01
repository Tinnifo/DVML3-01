# DNA Embedding Models - OUR and REVISIT

Two DNA sequence embedding models using k-mer representations:
- **OUR**: Supervised Contrastive Learning with multi-view representations
- **REVISIT**: Pairwise learning with negative sampling

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tinnifo/DVML3-01.git
   
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases** (optional, for experiment tracking):
   ```bash
   wandb login
   ```

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
