# This script has been adapted from the file available at the following address:
# https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/eval_binning.py
import csv
import argparse
import os
import sys
import collections
import numpy as np
import sklearn.metrics
from datetime import datetime
from evaluation.utils import align_labels_via_hungarian_algorithm
from evaluation.utils import (
    get_embedding,
    KMedoid,
    compute_class_center_medium_similarity,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)
MAX_SEQ_LEN = 20000
MIN_SEQ_LEN = 2500
MIN_ABUNDANCE_VALUE = 10


def main(args):
    model_list = args.model_list.split(",")
    for model_name in model_list:
        for species in args.species.split(","):
            for sample in map(int, args.samples.split(",")):
                # Define the appropriate metric for the given method
                # OUR and REVISIT models use L2 (Euclidean) distance
                if args.metric is not None:
                    metric = args.metric
                else:
                    metric = (
                        "l2"  # Default for nonlinear (REVISIT) and our (OUR) models
                    )

                print(
                    f"Model: {model_name} Species: {species} Sample ID: {sample} Metric: {metric}"
                )

                # Load the clustering data to compute similarity threshold
                clustering_data_file_path = os.path.join(
                    args.data_dir, species, f"clustering_0.tsv"
                )
                with open(clustering_data_file_path, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                # Shorten the sequences if they are longer than the
                dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                labels = [d[1] for d in data]

                # convert labels to numeric values
                label2id = {l: i for i, l in enumerate(set(labels))}
                labels = np.array([label2id[l] for l in labels])
                num_clusters = len(label2id)
                print(
                    f"Clustering data contains {len(dna_sequences)} sequences with {num_clusters} clusters."
                )

                # Get embeddings
                embedding = get_embedding(
                    dna_sequences=dna_sequences,
                    model_name=model_name,
                    species=species,
                    sample=0,
                    k=args.k,
                    task_name="clustering",
                    test_model_dir=args.test_model_dir,
                    suffix=args.suffix,
                )

                percentile_values = compute_class_center_medium_similarity(
                    embedding, labels, metric=metric
                )
                threshold = percentile_values[-3]
                print(f"Threshold value: {threshold}")

                # Load binning data
                data_file = os.path.join(
                    args.data_dir, species, f"binning_{sample}.tsv"
                )

                with open(data_file, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                labels_bin = [d[1] for d in data]

                # filter sequences with length < 2500
                filterd_idx = [
                    i for i, seq in enumerate(dna_sequences) if len(seq) >= MIN_SEQ_LEN
                ]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                # filter sequences with low abundance labels (less than 10)
                label_counts = collections.Counter(labels_bin)
                filterd_idx = [
                    i
                    for i, l in enumerate(labels_bin)
                    if label_counts[l] >= MIN_ABUNDANCE_VALUE
                ]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                # convert labels to numeric values
                label2id = {l: i for i, l in enumerate(set(labels_bin))}
                labels_bin = np.array([label2id[l] for l in labels_bin])
                num_clusters = len(label2id)
                print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")

                # Generate embeddings for the binning set
                embedding = get_embedding(
                    dna_sequences,
                    model_name,
                    species,
                    sample,
                    k=args.k,
                    metric=metric,
                    task_name="binning",
                    test_model_dir=args.test_model_dir,
                    suffix=args.suffix,
                )

                # Run the KMedoid algorithm
                binning_results = KMedoid(
                    embedding,
                    min_similarity=threshold,
                    min_bin_size=10,
                    max_iter=1000,
                    metric=metric,
                    scalable=args.scalable,
                )

                # Get the number of true labels and predictied labels
                true_labels_bin = labels_bin[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]
                print("Number of predicted labels: ", len(predicted_labels))

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(
                    true_labels_bin, predicted_labels
                )
                predicted_labels_bin = [
                    alignment_bin[label] for label in predicted_labels
                ]

                # Calculate purity, completeness, recall, and ARI
                recall_bin = sklearn.metrics.recall_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                recall_bin.sort()

                f1_bin = sklearn.metrics.f1_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                f1_bin.sort()
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                recall_results = []
                f1_results = []
                for threshold in thresholds:
                    recall_results.append(len(np.where(recall_bin > threshold)[0]))
                    f1_results.append(len(np.where(f1_bin > threshold)[0]))

                print(f"f1_results: {f1_results}")
                print(f"recall_results: {recall_results} \n")

                # Log to W&B if enabled
                if args.wandb_log and WANDB_AVAILABLE:
                    try:
                        # Initialize W&B if not already initialized
                        if not wandb.run:
                            wandb.init(
                                project=args.wandb_project if args.wandb_project else "dna-embedding-eval",
                                entity=args.wandb_entity,
                                mode=args.wandb_mode,
                                tags=["evaluation", "binning", model_name],
                            )
                        
                        eval_metrics = {
                            "eval/threshold": threshold,
                            "eval/num_clusters": num_clusters,
                            "eval/num_sequences": len(dna_sequences),
                            "eval/num_predicted": len(predicted_labels),
                            "eval/model": model_name,
                            "eval/species": species,
                            "eval/sample": sample,
                        }
                        
                        thresholds_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        for i, thresh in enumerate(thresholds_list):
                            eval_metrics[f"eval/recall_at_{thresh}"] = recall_results[i]
                            eval_metrics[f"eval/f1_at_{thresh}"] = f1_results[i]
                        
                        wandb.log(eval_metrics)
                        print(f"Evaluation metrics logged to W&B: {eval_metrics}")
                    except Exception as e:
                        print(f"Warning: Failed to log to W&B: {e}")

                with open(args.output, "a+") as f:
                    f.write("\n")
                    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    f.write(
                        f"model: {model_name}, species: {species}, sample: {sample}, binning\n"
                    )
                    f.write(f"recall_results: {recall_results}\n")
                    f.write(f"f1_results: {f1_results}\n")
                    f.write(f"threshold: {threshold}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering")
    parser.add_argument(
        "--species",
        type=str,
        default="reference,marine,plant",
        help="Species to evaluate",
    )
    parser.add_argument(
        "--samples", type=str, default="5,6", help="Species to evaluate"
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--test_model_dir",
        type=str,
        default="/root/trained_model",
        help="Path to trained model file (.pt) for nonlinear/our models",
    )
    parser.add_argument(
        "--model_list",
        type=str,
        default="nonlinear,our",
        help="List of models to evaluate, separated by comma. Supports [nonlinear (REVISIT), our (OUR/SupCon)]",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="k-mer size used in the model (should match training)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to measure the similarities among embeddings",
    )
    parser.add_argument(
        "--scalable",
        type=bool,
        default=0,
        help="Controls how we compute the similarity among embeddings",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to the output embedding file",
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
        help="W&B entity/team name",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    args = parser.parse_args()
    
    if args.wandb_log and not WANDB_AVAILABLE:
        print("Warning: wandb is not installed. W&B logging will be disabled.")
        print("Install with: pip install wandb")
        args.wandb_log = False
    
    main(args)
