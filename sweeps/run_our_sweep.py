#!/usr/bin/env python3
"""
Helper script to initialize and run W&B sweep for OUR.py (Hour Model).

Usage:
    python sweeps/run_our_sweep.py [--sweep-config sweeps/our_sweep.yaml] [--project PROJECT_NAME] [--count COUNT]

This script will:
1. Initialize a W&B sweep from the configuration file
2. Optionally run the sweep agent locally
"""

import argparse
import wandb
from wandb.errors import UsageError, AuthenticationError
import yaml
import os
import sys

# ============================================================================
# CONFIGURATION: Set your W&B entity here
# ============================================================================
WANDB_ENTITY = "tinnifo"
# ============================================================================


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def initialize_sweep(
    config_path: str, project: str = None, entity: str = None
) -> tuple:
    """
    Initialize a W&B sweep from configuration file.

    Returns:
        tuple: (sweep_id, actual_entity_used) - The ID of the created sweep and the entity that was actually used
    """
    config = load_sweep_config(config_path)

    # Override project if provided
    if project:
        # Note: W&B sweep config doesn't have project in the YAML,
        # it's set when initializing
        pass

    # Entity must be provided
    if entity is None:
        raise ValueError("Entity must be specified")

    print(f"Initializing sweep from {config_path}...")
    print(f"Using entity: {entity}, project: {project or 'dna-embedding-our'}")

    # Set entity in environment variable as wandb might use it
    os.environ["WANDB_ENTITY"] = entity
    if project:
        os.environ["WANDB_PROJECT"] = project

    try:
        # Try with entity parameter
        sweep_id = wandb.sweep(config, project=project, entity=entity)
        print(f"Sweep initialized with ID: {sweep_id}")
        print(
            f"View sweep at: https://wandb.ai/{entity}/{project or 'dna-embedding-our'}/sweeps/{sweep_id}"
        )
        return sweep_id, entity
    except UsageError as e:
        error_msg = str(e).lower()
        if (
            "user not valid" in error_msg
            or "sweep user not valid" in error_msg
            or "permission denied" in error_msg
        ):
            print(
                f"\nError: Permission denied for entity '{entity}' - you don't have permission to create sweeps under it."
            )
            print(f"Full error: {e}")
            print("\nTrying alternative approach using your logged-in entity...")

            # Try without entity parameter - let wandb use default (logged-in entity)
            try:
                # Remove entity from environment and try with just project
                if "WANDB_ENTITY" in os.environ:
                    del os.environ["WANDB_ENTITY"]
                sweep_id = wandb.sweep(config, project=project)

                # Get the actual entity that was used (logged-in entity)
                try:
                    api = wandb.Api()
                    viewer = api.viewer()
                    if viewer and "entity" in viewer:
                        actual_entity = viewer["entity"]
                    else:
                        actual_entity = None
                except Exception:
                    actual_entity = None

                if actual_entity:
                    print(
                        f"Sweep initialized with ID: {sweep_id} (using entity: {actual_entity})"
                    )
                    print(
                        f"View sweep at: https://wandb.ai/{actual_entity}/{project or 'dna-embedding-our'}/sweeps/{sweep_id}"
                    )
                else:
                    print(
                        f"Sweep initialized with ID: {sweep_id} (using your logged-in entity)"
                    )
                    print(
                        f"View sweep at: https://wandb.ai/{project or 'dna-embedding-our'}/sweeps/{sweep_id}"
                    )
                    # If we can't determine the entity, use None to let wandb figure it out
                    actual_entity = None

                return sweep_id, actual_entity
            except Exception as e2:
                print(f"\nAlternative approach also failed: {e2}")
                print("\nPossible solutions:")
                print(
                    "1. Update WANDB_ENTITY in the script to match your logged-in username"
                )
                print("2. Check your wandb profile to see your exact entity name")
                print(
                    "3. If using a team, ensure you're a member with write permissions"
                )
                print("4. Try using your personal username instead of a team name")
                raise e
        else:
            raise


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
    print("Press Ctrl+C to stop the agent")

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
        description="Initialize and run W&B sweep for OUR.py model"
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="sweeps/our_sweep.yaml",
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name (overrides config)",
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

    # Get entity from configuration
    entity = WANDB_ENTITY
    if entity is None:
        print("Error: WANDB_ENTITY is not set in the script.")
        print(
            "Please edit sweeps/run_our_sweep.py and set WANDB_ENTITY to your W&B entity/team name."
        )
        print("Example: WANDB_ENTITY = 'your-username'")
        sys.exit(1)

    print(f"Using W&B entity: {entity}")

    # Verify wandb login status and get logged-in entity
    logged_in_entity = None
    try:
        api = wandb.Api()
        viewer = api.viewer()
        if viewer and "entity" in viewer:
            logged_in_entity = viewer["entity"]
            print(f"Currently logged in as: {logged_in_entity}")
            if entity != logged_in_entity:
                print(
                    f"Warning: Using entity '{entity}' but logged in as '{logged_in_entity}'"
                )
                print(
                    "If you get a permission error, the script will automatically use your logged-in entity."
                )
        else:
            raise AuthenticationError("Could not get viewer information")
    except AuthenticationError:
        print("\nError: Not logged in to wandb or API key is invalid.")
        print("Please log in first by running: wandb login")
        print("Or set your API key: export WANDB_API_KEY=your_api_key")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not verify wandb login status: {e}")
        print("Continuing anyway, but you may encounter errors...")

    # If entity doesn't match logged-in entity, suggest updating it
    if logged_in_entity and entity != logged_in_entity:
        print(
            f"\nTip: Consider updating WANDB_ENTITY to '{logged_in_entity}' to avoid permission issues."
        )

    sweep_id, actual_entity = initialize_sweep(
        args.sweep_config,
        project=args.project or "dna-embedding-our",
        entity=entity,
    )

    # Use the actual entity that was used to create the sweep
    # If actual_entity is None, it means we used the logged-in entity but couldn't determine it
    # In that case, pass None to let wandb figure it out automatically
    entity_for_agent = actual_entity if actual_entity is not None else None

    # Run agent if not init-only
    if not args.init_only:
        run_sweep_agent(
            sweep_id,
            count=args.count,
            project=args.project or "dna-embedding-our",
            entity=entity_for_agent,
        )
    else:
        print("\nSweep initialized. Run the agent separately with:")
        if actual_entity:
            print(
                f"  wandb agent {actual_entity}/{args.project or 'dna-embedding-our'}/{sweep_id}"
            )
        else:
            print(
                f"  wandb agent {entity}/{args.project or 'dna-embedding-our'}/{sweep_id}"
            )


if __name__ == "__main__":
    main()
