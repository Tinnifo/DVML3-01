#!/usr/bin/env python3
"""
Script to download models from W&B sweeps and evaluate them.

Usage:
    python evaluation/evaluate_sweep_models.py \
        --sweep_id <sweep_id> \
        --entity <entity> \
        --project <project> \
        --data_dir <path_to_eval_data> \
        [--output <output_file>] \
        [--top_n <number_of_best_runs>] \
        [--species <species_list>] \
        [--samples <sample_list>] \
        [--k <kmer_size>]

Example:
    python evaluation/evaluate_sweep_models.py \
        --sweep_id t6bc1ftg \
        --entity tinnifo-projects \
        --project dna-embedding-our \
        --data_dir /path/to/evaluation/data \
        --top_n 5 \
        --species reference,marine,plant \
        --samples 5,6
"""

import argparse
import os
import sys
import wandb
from wandb import Api
import subprocess
import tempfile
import shutil


def download_model_from_run(run, download_dir):
    """Download model file from a W&B run."""
    model_files = []

    # Look for model files in the run's files
    for file in run.files():
        if file.name.endswith(".pt") or "model" in file.name.lower():
            try:
                downloaded_path = file.download(root=download_dir, replace=True)
                model_files.append(downloaded_path)
                print(f"  Downloaded: {file.name} -> {downloaded_path}")
            except Exception as e:
                print(f"  Warning: Failed to download {file.name}: {e}")

    return model_files


def get_best_runs(sweep_id, entity, project, top_n=5, metric="eval/f1_at_0.5"):
    """Get the top N runs from a sweep based on a metric."""
    api = Api()

    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs

        # Filter runs that have the metric
        runs_with_metric = []
        for run in runs:
            if run.summary.get(metric) is not None:
                runs_with_metric.append(run)

        # Sort by metric (descending)
        runs_with_metric.sort(
            key=lambda r: r.summary.get(metric, float("-inf")), reverse=True
        )

        print(f"\nFound {len(runs_with_metric)} runs with metric '{metric}'")
        print(f"Selecting top {min(top_n, len(runs_with_metric))} runs:")

        for i, run in enumerate(runs_with_metric[:top_n]):
            metric_value = run.summary.get(metric, "N/A")
            print(f"  {i + 1}. Run {run.id}: {metric} = {metric_value}")

        return runs_with_metric[:top_n]

    except Exception as e:
        print(f"Error accessing sweep: {e}")
        print(f"Trying to get all runs from sweep...")

        # Fallback: get all runs
        try:
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            runs = list(sweep.runs)
            print(f"Found {len(runs)} runs total")
            return runs[:top_n]
        except Exception as e2:
            print(f"Error: {e2}")
            return []


def evaluate_model(model_path, args):
    """Run evaluation for a single model."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating model: {model_path}")
    print(f"{'=' * 60}")

    # Build evaluation command
    eval_cmd = [
        sys.executable,
        "evaluation/binning.py",
        "--test_model_dir",
        model_path,
        "--model_list",
        "our",
        "--data_dir",
        args.data_dir,
        "--species",
        args.species,
        "--samples",
        args.samples,
        "--k",
        str(args.k),
    ]

    if args.output:
        eval_cmd.extend(["--output", args.output])

    if args.wandb_log:
        eval_cmd.append("--wandb_log")
        if args.wandb_project:
            eval_cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            eval_cmd.extend(["--wandb_entity", args.wandb_entity])
        if args.wandb_mode:
            eval_cmd.extend(["--wandb_mode", args.wandb_mode])

    if args.metric_eval:
        eval_cmd.extend(["--metric", args.metric_eval])

    if args.scalable:
        eval_cmd.append("--scalable")

    if args.suffix:
        eval_cmd.extend(["--suffix", args.suffix])

    print(f"Running: {' '.join(eval_cmd)}")

    # Run evaluation
    result = subprocess.run(eval_cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download models from W&B sweeps and evaluate them"
    )
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B sweep ID")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity name")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to evaluation data directory"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top runs to evaluate (default: 5)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/f1_at_0.5",
        help="Metric to use for ranking runs (default: eval/f1_at_0.5)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="reference,marine,plant",
        help="Comma-separated list of species to evaluate",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="5,6",
        help="Comma-separated list of sample IDs to evaluate",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="k-mer size used in the model (should match training)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for evaluation results"
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to download models (default: temporary directory)",
    )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        help="Enable W&B logging for evaluation metrics",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name for evaluation logging",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name for evaluation logging",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    parser.add_argument(
        "--metric_eval",
        type=str,
        default=None,
        help="Metric to measure similarities among embeddings",
    )
    parser.add_argument(
        "--scalable", action="store_true", help="Use scalable similarity computation"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix to add to output embedding file"
    )

    args = parser.parse_args()

    # Set up download directory
    if args.download_dir:
        download_dir = args.download_dir
        os.makedirs(download_dir, exist_ok=True)
        cleanup_download_dir = False
    else:
        download_dir = tempfile.mkdtemp(prefix="wandb_models_")
        cleanup_download_dir = True

    try:
        print(f"Download directory: {download_dir}")

        # Get best runs from sweep
        print(
            f"\nFetching runs from sweep: {args.entity}/{args.project}/{args.sweep_id}"
        )
        runs = get_best_runs(
            args.sweep_id, args.entity, args.project, args.top_n, args.metric
        )

        if not runs:
            print("No runs found. Trying to get all runs...")
            api = Api()
            sweep = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}")
            runs = list(sweep.runs)[: args.top_n]
            print(f"Found {len(runs)} runs")

        if not runs:
            print("Error: No runs found in sweep!")
            return 1

        # Download models and evaluate
        evaluated_count = 0
        for i, run in enumerate(runs):
            print(f"\n{'=' * 60}")
            print(f"Processing run {i + 1}/{len(runs)}: {run.id}")
            print(f"Run name: {run.name}")
            print(f"Config: {run.config}")
            print(f"{'=' * 60}")

            # Download model files
            model_files = download_model_from_run(run, download_dir)

            if not model_files:
                print(f"  Warning: No model files found for run {run.id}")
                print(f"  Skipping evaluation for this run")
                continue

            # Evaluate each model file
            for model_file in model_files:
                if evaluate_model(model_file, args):
                    evaluated_count += 1
                    print(f"  ✓ Evaluation completed successfully")
                else:
                    print(f"  ✗ Evaluation failed")

        print(f"\n{'=' * 60}")
        print(f"Summary: Evaluated {evaluated_count} model(s) from {len(runs)} run(s)")
        print(f"{'=' * 60}")

        return 0 if evaluated_count > 0 else 1

    finally:
        # Cleanup
        if cleanup_download_dir and os.path.exists(download_dir):
            print(f"\nCleaning up download directory: {download_dir}")
            shutil.rmtree(download_dir)


if __name__ == "__main__":
    sys.exit(main())
