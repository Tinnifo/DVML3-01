#!/usr/bin/env python3
"""
Visualization script for comparing OUR and REVISIT models on metagenomic binning task.
Creates stacked horizontal bar charts showing F1 score ranges similar to Figure 3.

After your sweeps are complete, simply run:

**If models are in different projects (default setup):**
```bash
python visualize_f1_ranges.py \
    --our_wandb_project dna-embedding-our \
    --revisit_wandb_project dna-embedding-revisit \
    --output f1_ranges_comparison.png
```

**If both models are in the same project:**
```bash
python visualize_f1_ranges.py \
    --wandb_project your-project-name \
    --output f1_ranges_comparison.png
```

Usage:
    python visualize_f1_ranges.py --wandb_project <project_name> [options]
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print(
        "Error: wandb is required for this script. Please install it: pip install wandb"
    )
    sys.exit(1)


def fetch_f1_counts_from_wandb(wandb_project, wandb_entity=None, model_tag="our"):
    """
    Fetch F1 count metrics from W&B runs.

    Returns:
        dict: {
            (species, sample): {
                'f1_at_0.1': count,
                'f1_at_0.2': count,
                ...
            }
        }
    """
    api = wandb.Api()
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    runs = api.runs(project_path)

    results = defaultdict(dict)

    print(f"  Scanning {len(list(runs))} runs...")

    for run in runs:
        # Check if this run has the right tag
        tags = run.tags or []
        tag_lower = [tag.lower() for tag in tags]
        if model_tag.lower() not in tag_lower:
            continue

        # Get summary metrics (final logged values)
        summary = run.summary

        # Look for eval metrics in summary
        for metric_name, value in summary.items():
            if metric_name.startswith("eval/") and "/f1_at_" in metric_name:
                # Parse: eval/{species}_sample{sample}/f1_at_{thresh}
                parts = metric_name.split("/")
                if len(parts) >= 3:
                    species_sample = parts[1]  # e.g., "synthetic_sample5"
                    f1_thresh = parts[2]  # e.g., "f1_at_0.5"

                    # Parse species and sample
                    if "_sample" in species_sample:
                        species = species_sample.split("_sample")[0]
                        try:
                            sample = int(species_sample.split("_sample")[1])

                            # Store the value
                            if value is not None and not (
                                isinstance(value, float) and np.isnan(value)
                            ):
                                key = (species, sample)
                                if key not in results:
                                    results[key] = {}
                                # Use the latest value if multiple runs have same key
                                if (
                                    f1_thresh not in results[key]
                                    or value > results[key][f1_thresh]
                                ):
                                    results[key][f1_thresh] = (
                                        int(value)
                                        if isinstance(value, (int, float))
                                        else value
                                    )
                        except (ValueError, IndexError):
                            continue

        # Also check history for metrics that might not be in summary
        try:
            history = run.history()
            if not history.empty:
                for metric_name in history.columns:
                    if metric_name.startswith("eval/") and "/f1_at_" in metric_name:
                        parts = metric_name.split("/")
                        if len(parts) >= 3:
                            species_sample = parts[1]
                            f1_thresh = parts[2]

                            if "_sample" in species_sample:
                                species = species_sample.split("_sample")[0]
                                try:
                                    sample = int(species_sample.split("_sample")[1])
                                    # Get the last non-null value
                                    values = history[metric_name].dropna()
                                    if len(values) > 0:
                                        value = values.iloc[-1]
                                        key = (species, sample)
                                        if key not in results:
                                            results[key] = {}
                                        if f1_thresh not in results[
                                            key
                                        ] or value > results[key].get(f1_thresh, 0):
                                            results[key][f1_thresh] = (
                                                int(value)
                                                if isinstance(value, (int, float))
                                                else value
                                            )
                                except (ValueError, IndexError):
                                    continue
        except Exception:
            # Some runs might not have history
            pass

    print(f"  Found results for {len(results)} species/sample combinations")
    return dict(results)


def reconstruct_f1_ranges_from_counts(f1_counts):
    """
    Reconstruct F1 range counts from threshold counts.

    The logged metrics are: f1_at_0.1, f1_at_0.2, ..., f1_at_0.9
    which represent the number of clusters with F1 > threshold.

    We need to compute:
    - (0.5, 0.6]: f1_at_0.5 - f1_at_0.6
    - (0.6, 0.7]: f1_at_0.6 - f1_at_0.7
    - etc.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ranges = {
        (0.5, 0.6): 0,
        (0.6, 0.7): 0,
        (0.7, 0.8): 0,
        (0.8, 0.9): 0,
        (0.9, 1.0): 0,
    }

    # Get counts at each threshold
    counts = {}
    for thresh in thresholds:
        key = f"f1_at_{thresh}"
        counts[thresh] = f1_counts.get(key, 0)

    # Reconstruct ranges
    # (0.5, 0.6] = clusters with F1 > 0.5 but <= 0.6 = f1_at_0.5 - f1_at_0.6
    ranges[(0.5, 0.6)] = max(0, counts[0.5] - counts[0.6])
    ranges[(0.6, 0.7)] = max(0, counts[0.6] - counts[0.7])
    ranges[(0.7, 0.8)] = max(0, counts[0.7] - counts[0.8])
    ranges[(0.8, 0.9)] = max(0, counts[0.8] - counts[0.9])
    # (0.9, 1.0] = clusters with F1 > 0.9
    ranges[(0.9, 1.0)] = counts[0.9]

    return ranges


def create_visualization(
    results,
    output_path="f1_ranges_comparison.png",
    title="Evaluation of the models on multiple datasets for the metagenomic binning task",
):
    """
    Create the stacked horizontal bar chart visualization.

    Args:
        results: dict with structure:
            {
                (species, sample): {
                    'OUR': {f1_range: count, ...},
                    'REVISIT': {f1_range: count, ...}
                }
            }
    """
    # Define F1 ranges and colors
    f1_ranges = [
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
    ]
    colors = {
        (0.5, 0.6): "#8B0000",  # Dark Red
        (0.6, 0.7): "#FF8C00",  # Orange
        (0.7, 0.8): "#ADFF2F",  # Light Green/Yellow
        (0.8, 0.9): "#00CED1",  # Light Blue/Cyan
        (0.9, 1.0): "#00008B",  # Dark Blue
    }

    # Organize data by dataset type and sample
    datasets = {}
    for (species, sample), model_results in results.items():
        if species not in datasets:
            datasets[species] = {}
        datasets[species][sample] = model_results

    # Determine which species and samples we have
    species_list = sorted(datasets.keys())
    sample_list = sorted(
        set(
            sample
            for species_data in datasets.values()
            for sample in species_data.keys()
        )
    )

    # Create figure with subplots
    n_species = len(species_list)
    n_samples = len(sample_list)

    fig, axes = plt.subplots(
        n_samples, n_species, figsize=(6 * n_species, 4 * n_samples)
    )
    # Handle different subplot configurations
    if n_samples == 1 and n_species == 1:
        # Single subplot - axes is a single Axes object
        axes = np.array([[axes]])
    elif n_samples == 1:
        # One row, multiple columns - axes is a 1D array
        axes = axes.reshape(1, -1) if hasattr(axes, "reshape") else np.array([axes])
    elif n_species == 1:
        # Multiple rows, one column - axes is a 1D array
        axes = (
            axes.reshape(-1, 1)
            if hasattr(axes, "reshape")
            else np.array([[ax] for ax in axes])
        )
    else:
        # Multiple rows and columns - axes is already 2D
        axes = np.array(axes)

    # Plot each combination
    for sample_idx, sample in enumerate(sample_list):
        for species_idx, species in enumerate(species_list):
            ax = axes[sample_idx, species_idx]

            if (species, sample) not in results:
                ax.text(
                    0.5,
                    0.5,
                    f"No data\n{species} sample {sample}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{species} {sample}")
                continue

            model_results = results[(species, sample)]
            models = ["OUR", "REVISIT"]
            y_positions = np.arange(len(models))

            # Plot stacked bars for each model
            for model in models:
                if model not in model_results:
                    continue

                model_data = model_results[model]
                bottom = 0

                for f1_range in f1_ranges:
                    count = model_data.get(f1_range, 0)
                    if count > 0:
                        ax.barh(
                            models.index(model),
                            count,
                            left=bottom,
                            color=colors[f1_range],
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        bottom += count

            # Set x-axis limit based on max value
            max_count = 0
            for m in models:
                if m in model_results:
                    model_total = sum(model_results[m].values())
                    max_count = max(max_count, model_total)

            if max_count > 0:
                ax.set_xlim(0, max_count * 1.1)
            else:
                # If no data, set a small default range
                ax.set_xlim(0, 10)

            ax.set_xlabel("Species count")
            ax.set_ylabel("Models")
            ax.set_title(f"{species} {sample}")
            ax.set_yticks(y_positions)
            ax.set_yticklabels(models)
            ax.grid(axis="x", alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(
            facecolor=colors[r], edgecolor="black", label=f"({r[0]}, {r[1]}]"
        )
        for r in f1_ranges
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(f1_ranges),
        title="F₁ Ranges",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize F1 score ranges for OUR vs REVISIT models from W&B results"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (if both models are in the same project)",
    )
    parser.add_argument(
        "--our_wandb_project",
        type=str,
        default=None,
        help="W&B project name for OUR model (overrides --wandb_project for OUR)",
    )
    parser.add_argument(
        "--revisit_wandb_project",
        type=str,
        default=None,
        help="W&B project name for REVISIT model (overrides --wandb_project for REVISIT)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional, applies to both projects if not specified separately)",
    )
    parser.add_argument(
        "--our_wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name for OUR model (overrides --wandb_entity for OUR)",
    )
    parser.add_argument(
        "--revisit_wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name for REVISIT model (overrides --wandb_entity for REVISIT)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="f1_ranges_comparison.png",
        help="Output path for visualization (default: f1_ranges_comparison.png)",
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=None,
        help="Species to include (default: all found in W&B)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Sample IDs to include (default: all found in W&B)",
    )

    args = parser.parse_args()

    if not WANDB_AVAILABLE:
        print("Error: wandb is required. Please install it: pip install wandb")
        return

    # Determine project names for each model
    our_project = args.our_wandb_project or args.wandb_project
    revisit_project = args.revisit_wandb_project or args.wandb_project

    if not our_project:
        print("Error: Must specify --wandb_project or --our_wandb_project")
        return
    if not revisit_project:
        print("Error: Must specify --wandb_project or --revisit_wandb_project")
        return

    # Determine entity names for each model
    our_entity = args.our_wandb_entity or args.wandb_entity
    revisit_entity = args.revisit_wandb_entity or args.wandb_entity

    print(f"OUR model project: {our_project}")
    if our_entity:
        print(f"OUR model entity: {our_entity}")
    print(f"REVISIT model project: {revisit_project}")
    if revisit_entity:
        print(f"REVISIT model entity: {revisit_entity}")

    # Fetch results for both models
    print("\nFetching OUR model results...")
    our_results = fetch_f1_counts_from_wandb(
        our_project, wandb_entity=our_entity, model_tag="our"
    )

    print("\nFetching REVISIT model results...")
    revisit_results = fetch_f1_counts_from_wandb(
        revisit_project, wandb_entity=revisit_entity, model_tag="revisit"
    )

    if not our_results and not revisit_results:
        print("Error: No results found in W&B. Please check:")
        print("  1. Project name is correct")
        print("  2. Runs have tags 'our' or 'revisit'")
        print(
            "  3. Runs have logged eval metrics (eval/{species}_sample{sample}/f1_at_*)"
        )
        return

    # Get all available species and samples
    all_keys = set(our_results.keys()) | set(revisit_results.keys())

    if args.species or args.samples:
        # Filter to requested species/samples
        filtered_keys = set()
        for species, sample in all_keys:
            if args.species and species not in args.species:
                continue
            if args.samples and sample not in args.samples:
                continue
            filtered_keys.add((species, sample))
        all_keys = filtered_keys

    if not all_keys:
        print("Error: No matching results found for specified species/samples.")
        return

    print(f"\nFound results for {len(all_keys)} species/sample combinations")

    # Reconstruct F1 ranges and organize results
    results = {}
    for species, sample in sorted(all_keys):
        our_f1_counts = our_results.get((species, sample), {})
        revisit_f1_counts = revisit_results.get((species, sample), {})

        if not our_f1_counts and not revisit_f1_counts:
            continue

        results[(species, sample)] = {
            "OUR": reconstruct_f1_ranges_from_counts(our_f1_counts)
            if our_f1_counts
            else {},
            "REVISIT": reconstruct_f1_ranges_from_counts(revisit_f1_counts)
            if revisit_f1_counts
            else {},
        }

        print(
            f"  {species} sample {sample}: OUR={bool(our_f1_counts)}, REVISIT={bool(revisit_f1_counts)}"
        )

    if not results:
        print("Error: No valid results to visualize.")
        return

    # Create visualization
    print(f"\n{'=' * 60}")
    print("Creating visualization...")
    print(f"{'=' * 60}")
    create_visualization(results, output_path=args.output)
    print(f"\n✓ Visualization complete! Saved to {args.output}")


if __name__ == "__main__":
    main()
