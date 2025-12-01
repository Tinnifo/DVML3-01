# Evaluating Models from W&B Sweeps

This guide explains how to evaluate models trained during W&B sweeps.

## Quick Start

### Option 1: Automatic Download and Evaluation (Recommended)

Use the `evaluate_sweep_models.py` script to automatically download models from your sweeps and evaluate them:

```bash
python evaluation/evaluate_sweep_models.py \
    --sweep_id <your_sweep_id> \
    --entity <your_entity> \
    --project dna-embedding-our \
    --data_dir <path_to_evaluation_data> \
    --top_n 5 \
    --species reference,marine,plant \
    --samples 5,6
```

**Example:**
```bash
python evaluation/evaluate_sweep_models.py \
    --sweep_id t6bc1ftg \
    --entity tinnifo-projects \
    --project dna-embedding-our \
    --data_dir /path/to/evaluation/data \
    --top_n 5 \
    --species reference \
    --samples 5
```

### Option 2: Manual Evaluation

If you already have model files locally:

```bash
python evaluation/binning.py \
    --test_model_dir <path_to_model.pt> \
    --model_list our \
    --data_dir <path_to_evaluation_data> \
    --species reference,marine,plant \
    --samples 5,6 \
    --k 4 \
    --output evaluation_results.txt
```

## Finding Your Sweep IDs

1. **From W&B Dashboard:**
   - Go to https://wandb.ai/<your-entity>/dna-embedding-our
   - Click on your sweep
   - The sweep ID is in the URL: `https://wandb.ai/<entity>/<project>/sweeps/<SWEEP_ID>`

2. **From Terminal Output:**
   - When you ran the sweep, it printed: `Sweep initialized with ID: <SWEEP_ID>`

## Parameters Explained

### Required Parameters

- `--sweep_id`: The W&B sweep ID (e.g., `t6bc1ftg`)
- `--entity`: Your W&B entity name (e.g., `tinnifo-projects`)
- `--project`: W&B project name (usually `dna-embedding-our`)
- `--data_dir`: Path to your evaluation data directory

### Optional Parameters

- `--top_n`: Number of best runs to evaluate (default: 5)
- `--metric`: Metric to use for ranking runs (default: `eval/f1_at_0.5`)
- `--species`: Comma-separated species list (default: `reference,marine,plant`)
- `--samples`: Comma-separated sample IDs (default: `5,6`)
- `--k`: k-mer size used in training (default: 4)
- `--output`: Output file for results (optional)
- `--download_dir`: Where to save downloaded models (default: temp directory)
- `--wandb_log`: Enable W&B logging for evaluation metrics

## Evaluation Data Structure

Your evaluation data directory should have this structure:

```
evaluation_data/
├── reference/
│   ├── clustering_0.tsv
│   ├── binning_5.tsv
│   └── binning_6.tsv
├── marine/
│   ├── clustering_0.tsv
│   ├── binning_5.tsv
│   └── binning_6.tsv
└── plant/
    ├── clustering_0.tsv
    ├── binning_5.tsv
    └── binning_6.tsv
```

## Output

The evaluation script will:
1. Download model files from W&B runs
2. Run binning evaluation for each model
3. Calculate F1 scores and recall at various thresholds (0.1-0.9)
4. Optionally log results to W&B

Results are printed to console and optionally saved to a file if `--output` is specified.

## Troubleshooting

### No models found in runs
- Check that runs completed successfully
- Models are saved with `.pt` extension
- Check W&B run artifacts/files tab

### Evaluation fails
- Ensure evaluation data directory exists and has correct structure
- Check that `--k` parameter matches the k-mer size used during training
- Verify model file is valid PyTorch checkpoint

### Permission errors
- Make sure you're logged in: `wandb login`
- Verify you have access to the sweep entity/project

## Examples

### Evaluate top 3 runs from a sweep:
```bash
python evaluation/evaluate_sweep_models.py \
    --sweep_id t6bc1ftg \
    --entity tinnifo-projects \
    --project dna-embedding-our \
    --data_dir ./evaluation_data \
    --top_n 3 \
    --output top3_results.txt
```

### Evaluate with W&B logging:
```bash
python evaluation/evaluate_sweep_models.py \
    --sweep_id t6bc1ftg \
    --entity tinnifo-projects \
    --project dna-embedding-our \
    --data_dir ./evaluation_data \
    --wandb_log \
    --wandb_project dna-embedding-eval \
    --wandb_entity tinnifo-projects
```

### Evaluate specific species and samples:
```bash
python evaluation/evaluate_sweep_models.py \
    --sweep_id t6bc1ftg \
    --entity tinnifo-projects \
    --project dna-embedding-our \
    --data_dir ./evaluation_data \
    --species reference \
    --samples 5
```

