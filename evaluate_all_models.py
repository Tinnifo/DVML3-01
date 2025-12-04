#!/usr/bin/env python3
"""
Evaluate all trained models and link results to W&B training runs.

This script implements Approach 3: Post-hoc optimization
1. Finds all models in a directory
2. Evaluates each model
3. Links evaluation results to original training runs in W&B
4. Enables analysis of which training hyperparameters led to best evaluation scores

Usage:
    # Evaluate all models and link to training runs
    python evaluate_all_models.py --model_dir models/ --model_type our \
        --eval_data_dir . --wandb_project dna-embedding-our --wandb_entity your-entity
    
    # Evaluate specific models by pattern
    python evaluate_all_models.py --model_dir models/ --model_pattern "our_model*.pt" \
        --model_type our --eval_data_dir . --wandb_project dna-embedding-our
"""

import argparse
import os
import sys
import glob
import wandb
from wandb import Api
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluate_models import run_evaluation


def find_training_run_for_model(model_path, wandb_project, wandb_entity=None):
    """
    Find the W&B training run that created this model.
    
    Args:
        model_path: Path to the model file
        wandb_project: W&B project name
        wandb_entity: W&B entity name (optional)
    
    Returns:
        run_id: W&B run ID if found, None otherwise
    """
    try:
        api = Api()
        project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
        
        # Search for runs in the project
        runs = api.runs(project_path)
        
        # Try to match by model path in summary
        model_basename = os.path.basename(model_path)
        model_abs_path = os.path.abspath(model_path)
        
        for run in runs:
            try:
                # Check if run summary has model_path
                if hasattr(run.summary, '_json_dict'):
                    summary = run.summary._json_dict
                else:
                    summary = dict(run.summary)
                
                if "model_path" in summary:
                    saved_path = summary["model_path"]
                    # Check if paths match (handle relative/absolute)
                    if (os.path.basename(saved_path) == model_basename or 
                        os.path.abspath(saved_path) == model_abs_path):
                        return run.id
                
                # Also check artifacts
                for artifact in run.logged_artifacts():
                    if artifact.type == "model":
                        if model_basename in str(artifact):
                            return run.id
            except Exception:
                continue
        
        # If not found by path, try to match by filename pattern
        # Extract hyperparameters from filename if possible
        # (e.g., "our_model_epoch=10_LR=0.0001.pt")
        return None
        
    except Exception as e:
        print(f"Warning: Could not search W&B runs: {e}")
        return None


def evaluate_all_models(
    model_dir,
    model_type,
    eval_config,
    wandb_config,
    model_pattern="*.pt",
    link_to_training=True,
):
    """
    Evaluate all models in a directory and optionally link to training runs.
    
    Args:
        model_dir: Directory containing model files
        model_type: 'our' or 'revisit'
        eval_config: Evaluation configuration
        wandb_config: W&B configuration
        model_pattern: Glob pattern to match model files
        link_to_training: Whether to link evaluation results to training runs
    """
    # Find all model files
    model_files = glob.glob(os.path.join(model_dir, model_pattern))
    model_files.sort()
    
    if not model_files:
        print(f"No model files found matching pattern '{model_pattern}' in {model_dir}")
        return
    
    print(f"Found {len(model_files)} model(s) to evaluate")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Processing: {os.path.basename(model_path)}")
        
        # Find corresponding training run if linking enabled
        training_run_id = None
        if link_to_training and wandb_config and wandb_config.get("project"):
            training_run_id = find_training_run_for_model(
                model_path,
                wandb_config["project"],
                wandb_config.get("entity"),
            )
            if training_run_id:
                print(f"  Linked to training run: {training_run_id}")
            else:
                print(f"  No training run found (will create new evaluation run)")
        
        # Prepare W&B config for this evaluation
        eval_wandb_config = wandb_config.copy() if wandb_config else None
        
        # If we found a training run, we can either:
        # 1. Log to the same run (update it with evaluation metrics)
        # 2. Create a new run and link it
        
        # For now, create a new run but tag it with training run ID
        if training_run_id and eval_wandb_config:
            # Add training run ID as tag for linking
            if "tags" not in eval_wandb_config:
                eval_wandb_config["tags"] = []
            eval_wandb_config["tags"].append(f"training_run_{training_run_id}")
            eval_wandb_config["tags"].append("post_hoc_evaluation")
        
        # Run evaluation
        try:
            eval_metrics = run_evaluation(
                model_path=model_path,
                model_type=model_type,
                eval_config=eval_config,
                wandb_config=eval_wandb_config,
            )
            
            if eval_metrics:
                results.append({
                    "model_path": model_path,
                    "training_run_id": training_run_id,
                    "eval_f1": eval_metrics.get("eval/f1_at_0.5"),
                    "eval_recall": eval_metrics.get("eval/recall_at_0.5"),
                })
        except Exception as e:
            print(f"  Error evaluating {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total models evaluated: {len(results)}/{len(model_files)}")
    
    if results:
        # Sort by F1 score
        results.sort(key=lambda x: x["eval_f1"] or 0, reverse=True)
        
        print(f"\nTop 5 models by F1@0.5:")
        for i, result in enumerate(results[:5], 1):
            print(f"  {i}. {os.path.basename(result['model_path'])}")
            print(f"     F1@0.5: {result['eval_f1']:.4f}, Recall@0.5: {result['eval_recall']:.4f}")
            if result['training_run_id']:
                print(f"     Training run: {result['training_run_id']}")
        
        print(f"\n{'='*60}")
        print("Next steps:")
        print("1. Go to W&B project to see all evaluation results")
        print("2. Use W&B's parallel coordinates or hyperparameter importance")
        print("3. Identify which training hyperparameters led to best eval/f1_at_0.5")
        print("4. Run another training sweep focusing on promising hyperparameters")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models and link to W&B training runs"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model files (.pt)",
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        default="*.pt",
        help="Glob pattern to match model files (default: *.pt)",
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
        help="Species for evaluation (comma-separated). Auto-detected if not specified.",
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
        required=True,
        help="W&B project name (must match training project to link runs)",
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
    
    parser.add_argument(
        "--no_link",
        action="store_true",
        help="Don't try to link evaluation results to training runs",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Prepare evaluation config
    eval_config = {
        "data_dir": args.eval_data_dir,
        "species": args.eval_species,
        "sample": args.eval_sample,
        "k": args.k,
        "metric": args.metric,
    }
    
    # Prepare W&B config
    wandb_config = None
    if args.wandb_mode != "disabled":
        wandb_config = {
            "enabled": True,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
        }
    
    # Evaluate all models
    evaluate_all_models(
        model_dir=args.model_dir,
        model_type=args.model_type,
        eval_config=eval_config,
        wandb_config=wandb_config,
        model_pattern=args.model_pattern,
        link_to_training=not args.no_link,
    )


if __name__ == "__main__":
    main()

