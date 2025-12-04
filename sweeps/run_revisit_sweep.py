#!/usr/bin/env python3
"""
Helper script to initialize and run W&B sweep for REVISIT.py (Revisit Model).

Usage:
    python sweeps/run_revisit_sweep.py [--sweep-config sweeps/revisit_sweep.yaml] [--project PROJECT_NAME] [--entity ENTITY_NAME] [--count COUNT]

This script will:
1. Initialize a W&B sweep from the configuration file
2. Optionally run the sweep agent locally
"""

import argparse
import wandb
import yaml
import os
import sys

# ============================================================================
# CONFIGURATION: Set your W&B entity here
# ============================================================================
WANDB_ENTITY = "tinnifo-aalborg-universitet"
# ============================================================================


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def initialize_sweep(config_path: str, project: str = None, entity: str = None) -> str:
    """
    Initialize a W&B sweep from configuration file.

    Returns:
        sweep_id: The ID of the created sweep
    """
    config = load_sweep_config(config_path)

    # Override project if provided
    if project:
        # Note: W&B sweep config doesn't have project in the YAML,
        # it's set when initializing
        pass

    print(f"Initializing sweep from {config_path}...")
    sweep_id = wandb.sweep(config, project=project, entity=entity)
    print(f"Sweep initialized with ID: {sweep_id}")
    print(
        f"View sweep at: https://wandb.ai/{entity or wandb.api.viewer()['entity']}/{project or 'dna-embedding-revisit'}/sweeps/{sweep_id}"
    )

    return sweep_id


def run_sweep_agent(
    sweep_id: str, count: int = None, project: str = None, entity: str = None
):
    """
    Run W&B sweep agent locally.

    Args:
        sweep_id: The sweep ID to run
        count: Number of runs to execute (None for unlimited)
        project: W&B project name
        entity: W&B entity name
    """
    print(f"Starting sweep agent for sweep {sweep_id}...")
    print(f"Press Ctrl+C to stop the agent")

    if count:
        print(f"Will run {count} training runs")

    wandb.agent(
        sweep_id,
        project=project,
        entity=entity,
        count=count,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Initialize and run W&B sweep for REVISIT.py model"
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="sweeps/revisit_sweep.yaml",
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name (overrides config)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/team name",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (None for unlimited)",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize the sweep, don't run the agent",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.sweep_config):
        print(f"Error: Sweep config file not found: {args.sweep_config}")
        sys.exit(1)

    # Initialize sweep
    sweep_id = initialize_sweep(
        args.sweep_config,
        project=args.project or "dna-embedding-revisit",
        entity=args.entity or WANDB_ENTITY,
    )

    # Run agent if not init-only
    if not args.init_only:
        run_sweep_agent(
            sweep_id,
            count=args.count,
            project=args.project or "dna-embedding-revisit",
            entity=args.entity or WANDB_ENTITY,
        )
    else:
        print("\nSweep initialized. Run the agent separately with:")
        print(
            f"  wandb agent {args.entity or WANDB_ENTITY}/{args.project or 'dna-embedding-revisit'}/{sweep_id}"
        )


if __name__ == "__main__":
    main()
