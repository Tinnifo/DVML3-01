#!/usr/bin/env python3
"""
Standalone script to evaluate trained models.

Usage:
    # Evaluate a single model
    python evaluate_models.py --model_path models/our_model.pt --model_type our --eval_data_dir .
    
    # Evaluate multiple models from a directory
    python evaluate_models.py --model_dir models/ --model_type our --eval_data_dir .
    
    # Evaluate with W&B logging
    python evaluate_models.py --model_path models/our_model.pt --model_type our --eval_data_dir . --wandb_project my-project --wandb_entity my-entity
    
    # Use with W&B sweeps (see sweeps/evaluation_sweep.yaml)
    # The sweep will optimize based on eval/f1_at_0.5 metric
"""

import argparse
import os
import sys
import torch
import glob
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import evaluation utilities
try:
    from evaluation.utils import (
        get_embedding,
        KMedoid,
        align_labels_via_hungarian_algorithm,
    )
    import sklearn.metrics
    import collections
    import csv
except ImportError as e:
    print(f"Error: Could not import evaluation modules: {e}")
    sys.exit(1)

# Import model classes
try:
    from src.OUR import NonLinearModel as OUR_NonLinearModel
    from src.REVISIT import NonLinearModel as REVISIT_NonLinearModel
except ImportError as e:
    print(f"Error: Could not import model classes: {e}")
    sys.exit(1)


def run_evaluation(
    model_path: str,
    model_type: str,
    eval_config: dict,
    wandb_config: dict = None,
):
    """
    Run evaluation on a single model.
    
    Args:
        model_path: Path to the model file
        model_type: 'our' or 'revisit'
        eval_config: Dictionary with evaluation configuration
        wandb_config: Optional W&B configuration for logging
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"{'='*60}\n")
    
    # Initialize W&B if configured
    wandb_initialized_by_us = False
    if wandb_config and wandb_config.get("enabled", False):
        import wandb
        # Check if wandb is already initialized (e.g., by a sweep)
        if wandb.run is None:
            wandb.init(
                project=wandb_config.get("project", "model-evaluation"),
                entity=wandb_config.get("entity", None),
                mode=wandb_config.get("mode", "online"),
                config={
                    "model_path": model_path,
                    "model_type": model_type,
                    **eval_config,
                },
                tags=["evaluation", model_type],
            )
            wandb_initialized_by_us = True
        else:
            # Already initialized (e.g., by sweep), just update config
            wandb.config.update({
                "model_path": model_path,
                "model_type": model_type,
                **eval_config,
            })
    
    data_dir = eval_config.get("data_dir")
    species_list = eval_config.get("species", "reference")
    if isinstance(species_list, str):
        if "," in species_list:
            species_list = [s.strip() for s in species_list.split(",")]
        else:
            species_list = [species_list]
    
    sample_list = eval_config.get("sample", [5])
    if isinstance(sample_list, (int, str)):
        if isinstance(sample_list, str) and "," in sample_list:
            sample_list = [int(s.strip()) for s in sample_list.split(",")]
        else:
            sample_list = [int(sample_list)]
    elif isinstance(sample_list, list):
        sample_list = [int(s) for s in sample_list]
    
    k = eval_config.get("k", 4)
    metric = eval_config.get("metric", "l2")
    
    if data_dir is None:
        print("Error: Evaluation data_dir not provided")
        return None
    
    if not os.path.exists(data_dir):
        print(f"Error: Evaluation data directory not found: {data_dir}")
        return None
    
    # Auto-detect available species
    available_species = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and os.path.exists(
            os.path.join(item_path, "clustering_0.tsv")
        ):
            available_species.append(item)
    
    if available_species:
        if not species_list:
            species_list = available_species
            print(f"Auto-detected species: {species_list}")
        else:
            species_list = [s for s in species_list if s in available_species]
            print(f"Evaluating species: {species_list}")
    
    # Evaluate all species and samples
    all_eval_metrics = {}
    f1_at_05_values = []
    recall_at_05_values = []
    
    for species in species_list:
        for sample in sample_list:
            try:
                print(f"\nEvaluating: species={species}, sample={sample}")
                
                # Load clustering data
                clustering_data_file_path = os.path.join(
                    data_dir, species, "clustering_0.tsv"
                )
                if not os.path.exists(clustering_data_file_path):
                    print(
                        f"Warning: Clustering data file not found: {clustering_data_file_path}"
                    )
                    continue
                
                with open(clustering_data_file_path, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]
                
                MAX_SEQ_LEN = 20000
                dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                labels = [d[1] for d in data]
                
                # Convert labels to numeric values
                unique_labels = sorted(list(set(labels)))
                label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                true_labels = [label_to_id[label] for label in labels]
                
                # Get embeddings
                model_name = "our" if model_type == "our" else "nonlinear"
                embedding = get_embedding(
                    dna_sequences,
                    model_name=model_name,
                    species=species,
                    sample=sample,
                    k=k,
                    metric=metric,
                    task_name="clustering",
                    test_model_dir=model_path,
                    suffix="",
                )
                
                # Compute similarity threshold from class centers
                from evaluation.utils import compute_class_center_medium_similarity
                percentile_values = compute_class_center_medium_similarity(
                    embedding, true_labels, metric=metric
                )
                threshold = percentile_values[4]  # 50th percentile (median)
                
                # Clustering
                predicted_labels = KMedoid(
                    embedding,
                    min_similarity=threshold,
                    min_bin_size=100,
                    max_iter=300,
                    metric=metric,
                    scalable=True,
                )
                
                # Filter out unassigned sequences (label -1)
                valid_mask = predicted_labels != -1
                if not any(valid_mask):
                    print(f"Warning: No valid clusters found for {species} sample {sample}")
                    continue
                
                filtered_true_labels = [true_labels[i] for i in range(len(true_labels)) if valid_mask[i]]
                filtered_predicted_labels = [predicted_labels[i] for i in range(len(predicted_labels)) if valid_mask[i]]
                
                # Align labels using Hungarian algorithm
                label_mapping = align_labels_via_hungarian_algorithm(
                    filtered_true_labels, filtered_predicted_labels
                )
                aligned_predicted_labels = [
                    label_mapping.get(pred, -1) for pred in filtered_predicted_labels
                ]
                
                # Compute metrics at different thresholds
                thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                recall_results = []
                f1_results = []
                
                for thresh in thresholds:
                    # Filter predictions by threshold
                    thresh_predicted = [
                        aligned_predicted_labels[i]
                        for i in range(len(aligned_predicted_labels))
                        if aligned_predicted_labels[i] != -1
                    ]
                    thresh_true = [
                        filtered_true_labels[i]
                        for i in range(len(aligned_predicted_labels))
                        if aligned_predicted_labels[i] != -1
                    ]
                    
                    if len(thresh_predicted) == 0:
                        recall_results.append(0.0)
                        f1_results.append(0.0)
                        continue
                    
                    recall = sklearn.metrics.recall_score(
                        thresh_true, thresh_predicted, average="macro", zero_division=0
                    )
                    f1 = sklearn.metrics.f1_score(
                        thresh_true, thresh_predicted, average="macro", zero_division=0
                    )
                    recall_results.append(recall)
                    f1_results.append(f1)
                
                # Count clusters
                num_clusters_bin = len(set([l for l in predicted_labels if l != -1]))
                
                # Store metrics
                prefix = f"eval/{species}_sample{sample}"
                all_eval_metrics[f"{prefix}/threshold"] = threshold
                all_eval_metrics[f"{prefix}/num_clusters"] = num_clusters_bin
                all_eval_metrics[f"{prefix}/num_sequences"] = len(dna_sequences)
                all_eval_metrics[f"{prefix}/num_predicted"] = len(filtered_predicted_labels)
                
                for i, thresh in enumerate(thresholds):
                    all_eval_metrics[f"{prefix}/recall_at_{thresh}"] = recall_results[i]
                    all_eval_metrics[f"{prefix}/f1_at_{thresh}"] = f1_results[i]
                
                # Store F1@0.5 and recall@0.5 for aggregation
                f1_at_05_idx = thresholds.index(0.5)
                f1_at_05_values.append(f1_results[f1_at_05_idx])
                recall_at_05_values.append(recall_results[f1_at_05_idx])
                
                print(f"  F1@0.5: {f1_results[f1_at_05_idx]:.4f}")
                print(f"  Recall@0.5: {recall_results[f1_at_05_idx]:.4f}")
                print(f"  Num clusters: {num_clusters_bin}")
                
            except Exception as e:
                print(f"Error evaluating {species} sample {sample}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate metrics
    if f1_at_05_values:
        all_eval_metrics["eval/f1_at_0.5"] = np.mean(f1_at_05_values)
        all_eval_metrics["eval/recall_at_0.5"] = np.mean(recall_at_05_values)
        all_eval_metrics["eval/f1_at_0.5_std"] = np.std(f1_at_05_values)
        all_eval_metrics["eval/recall_at_0.5_std"] = np.std(recall_at_05_values)
    
    # Log to W&B if configured
    if wandb_config and wandb_config.get("enabled", False):
        import wandb
        if all_eval_metrics:
            # Log all metrics (for time series)
            wandb.log(all_eval_metrics)
            # Update summary (for sweep optimization - these are the metrics sweeps use)
            wandb.run.summary.update(all_eval_metrics)
            # Explicitly set the primary metric for sweep optimization
            if "eval/f1_at_0.5" in all_eval_metrics:
                wandb.run.summary["eval/f1_at_0.5"] = all_eval_metrics["eval/f1_at_0.5"]
            print(f"\n✓ All evaluation metrics logged to W&B")
            print(f"  Total metrics logged: {len(all_eval_metrics)}")
            print(f"  Primary metric (for sweeps): eval/f1_at_0.5 = {all_eval_metrics.get('eval/f1_at_0.5', 'N/A')}")
        else:
            wandb.log({"evaluation_warning": "No evaluation metrics were generated"})
        
        # Only finish if we initialized wandb ourselves (not if it was initialized by a sweep)
        if wandb_initialized_by_us:
            wandb.finish()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print(f"{'='*60}")
    if all_eval_metrics:
        print(f"Mean F1@0.5: {all_eval_metrics.get('eval/f1_at_0.5', 'N/A'):.4f}")
        print(f"Mean Recall@0.5: {all_eval_metrics.get('eval/recall_at_0.5', 'N/A'):.4f}")
    else:
        print("No evaluation metrics generated")
    print(f"{'='*60}\n")
    
    return all_eval_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models (OUR or REVISIT)"
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Path to a single model file (.pt)",
    )
    model_group.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing model files (.pt) to evaluate",
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["our", "revisit"],
        help="Type of model: 'our' or 'revisit'",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        required=True,
        help="Data directory for evaluation (contains reference/, marine/, plant/ subdirectories)",
    )
    parser.add_argument(
        "--eval_species",
        type=str,
        default=None,
        help="Species for evaluation (comma-separated, e.g., 'reference,marine,plant'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--eval_sample",
        type=str,
        default="5,6",
        help="Sample ID(s) for evaluation (comma-separated, e.g., '5,6')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="k-mer size (default: 4)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["dot", "l2", "euclidean", "l1"],
        help="Similarity metric (default: l2)",
    )
    
    # W&B configuration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (if provided, results will be logged to W&B)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode (default: online)",
    )
    
    args = parser.parse_args()
    
    # Prepare evaluation config
    eval_config = {
        "data_dir": args.eval_data_dir,
        "species": args.eval_species,
        "sample": args.eval_sample,
        "k": args.k,
        "metric": args.metric,
    }
    
    # Prepare W&B config
    # Check if wandb is already initialized (e.g., by a sweep)
    try:
        import wandb
        wandb_initialized = wandb.run is not None
    except ImportError:
        wandb_initialized = False
        wandb = None
    
    wandb_config = None
    if wandb_initialized or (args.wandb_project is not None and args.wandb_mode != "disabled"):
        wandb_config = {
            "enabled": True,
            "project": args.wandb_project if not wandb_initialized else None,  # Use sweep's project if already initialized
            "entity": args.wandb_entity if not wandb_initialized else None,  # Use sweep's entity if already initialized
            "mode": args.wandb_mode if not wandb_initialized else "online",  # Use sweep's mode if already initialized
        }
    
    # Collect model paths
    model_paths = []
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            sys.exit(1)
        model_paths = [args.model_path]
    elif args.model_dir:
        if not os.path.exists(args.model_dir):
            print(f"Error: Model directory not found: {args.model_dir}")
            sys.exit(1)
        model_paths = glob.glob(os.path.join(args.model_dir, "*.pt"))
        if not model_paths:
            print(f"Error: No .pt model files found in {args.model_dir}")
            sys.exit(1)
        model_paths.sort()
    
    print(f"Found {len(model_paths)} model(s) to evaluate")
    
    # Evaluate each model
    all_results = {}
    for model_path in model_paths:
        try:
            results = run_evaluation(
                model_path=model_path,
                model_type=args.model_type,
                eval_config=eval_config,
                wandb_config=wandb_config,
            )
            all_results[model_path] = results
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete: {len(all_results)}/{len(model_paths)} models evaluated")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

