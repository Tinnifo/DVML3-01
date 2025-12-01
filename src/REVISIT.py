import torch
import random
import itertools
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import re
import warnings
import wandb
import os

# Suppress urllib3 OpenSSL warning on macOS (harmless, just noise)
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
except (ImportError, AttributeError):
    pass


def set_seed(seed):
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)


class PairDataset(Dataset):
    def __init__(
        self,
        file_path,
        transform_func,
        neg_sample_per_pos=1000,
        max_read_num=0,
        verbose=True,
        seed=0,
    ):
        # Set the parameters
        self.__both_kmer_profiles = None
        self.__transform_func = transform_func
        self.__neg_sample_per_pos = neg_sample_per_pos
        self.__seed = seed

        # Set the seed
        set_seed(seed)

        # Get the number of lines
        with open(file_path, "r") as f:
            lines_num = sum(1 for _ in f)
        # If the max_read_num is set, then sample the line number to read
        if max_read_num > 0:
            sample_size = min(max_read_num, lines_num)
            chosen_lines = random.sample(range(lines_num), sample_size)
            chosen_lines.sort()

        # Read the file
        chosen_line_idx = 0
        left_kmer_profiles, right_kmer_profiles = [], []
        with open(file_path, "r") as f:
            for current_line_idx, line in enumerate(f):
                if max_read_num > 0:
                    if chosen_line_idx == len(chosen_lines):
                        break

                    if current_line_idx != chosen_lines[chosen_line_idx]:
                        continue
                    else:
                        chosen_line_idx += 1

                # Remove the newline character and commas
                left_read, right_read = line.strip().split(",")
                left_kmer_profiles.append(self.__transform_func(left_read))
                right_kmer_profiles.append(self.__transform_func(right_read))

        # Combine the left and right k-mer profiles
        self.__both_kmer_profiles = torch.from_numpy(
            np.asarray(left_kmer_profiles + right_kmer_profiles)
        ).to(torch.float)

        if verbose:
            print(f"The data file was read successfully!")
            print(f"\t+ Total number of read pairs: {lines_num}")
            if max_read_num > 0:
                print(f"\t+ Number of read pairs used: {max_read_num}")

        # Temporary variables
        self.__ones = torch.ones((len(self.__both_kmer_profiles),))

    def __len__(self):
        return len(self.__both_kmer_profiles) // 2

    def __getitem__(self, idx):
        # Sample negative sample_indices
        negative_sample_indices = torch.multinomial(
            self.__ones, replacement=True, num_samples=2 * self.__neg_sample_per_pos
        )

        # Define the positive and negative k-mer profile pairs
        left_kmer_profiles = torch.concatenate(
            (
                self.__both_kmer_profiles[idx].unsqueeze(0),
                self.__both_kmer_profiles[
                    negative_sample_indices[: self.__neg_sample_per_pos]
                ],
            )
        )
        right_kmer_profiles = torch.concatenate(
            (
                self.__both_kmer_profiles[idx + self.__len__()].unsqueeze(0),
                self.__both_kmer_profiles[
                    negative_sample_indices[self.__neg_sample_per_pos :]
                ],
            )
        )
        # Define the labels
        labels = torch.tensor([1] + [0] * self.__neg_sample_per_pos, dtype=torch.float)

        return left_kmer_profiles, right_kmer_profiles, labels


class NonLinearModel(torch.nn.Module):
    def __init__(self, k, dim=256, device=torch.device("cpu"), verbose=False, seed=0):
        super(NonLinearModel, self).__init__()

        # Set the parameters
        self.__device = device
        self.__verbose = verbose

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        self.__dim = dim
        self.__letters = ["A", "C", "G", "T"]
        self.__kmer2id = {
            "".join(kmer): i
            for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))
        }
        self.__kmers_num = len(self.__kmer2id)

        # Set the seed
        set_seed(seed)

        # Define the layers
        self.linear1 = torch.nn.Linear(
            self.__kmers_num, 512, dtype=torch.float, device=self.__device
        )
        self.batch1 = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self.__device)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(
            512, self.__dim, dtype=torch.float, device=self.__device
        )

        self.bce_loss = torch.nn.BCELoss()

    def encoder(self, kmer_profile):
        output = self.linear1(kmer_profile)
        output = self.batch1(output)
        output = self.activation1(output)
        output = self.dropout1(output)
        output = self.linear2(output)

        return output

    def forward(self, left_kmer_profile, right_kmer_profile):
        left_output = self.encoder(left_kmer_profile)
        right_output = self.encoder(right_kmer_profile)

        return left_output, right_output

    def get_k(self):
        return self.__k

    def get_dim(self):
        return self.__dim

    def get_device(self):
        return self.__device

    def read2kmer_profile(self, read, normalized=True):
        # Get the k-mer profile
        kmer2id = [
            self.__kmer2id[read[i : i + self.__k]]
            for i in range(len(read) - self.__k + 1)
        ]
        kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_profile = kmer_profile / kmer_profile.sum()

        return kmer_profile

    def read2emb(self, reads, normalized=True):
        with torch.no_grad():
            kmer_profiles = []
            for read in reads:
                kmer_profiles.append(
                    self.read2kmer_profile(read, normalized=normalized)
                )

            kmer_profiles = torch.from_numpy(np.asarray(kmer_profiles)).to(torch.float)
            embs = self.encoder(kmer_profiles).detach().numpy()

        return embs


def loss_func(left_embeddings, right_embeddings, labels, name="bern"):
    if name == "bern":
        p = torch.exp(
            -(torch.norm(left_embeddings - right_embeddings, p=2, dim=1) ** 2)
        )

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction="mean")

    elif name == "poisson":
        log_lambda = -(torch.norm(left_embeddings - right_embeddings, p=2, dim=1) ** 2)

        return torch.mean(-(labels * log_lambda) + torch.exp(log_lambda))

    elif name == "hinge":
        d = torch.norm(left_embeddings - right_embeddings, p=2, dim=1)
        return torch.mean(
            labels * (d**2) + (1 - labels) * torch.nn.functional.relu(1 - d) ** 2
        )

    else:
        raise ValueError(f"Unknown loss function: {name}")


def single_epoch(model, loss_func, optimizer, training_loader, loss_name="bern"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch_loss = 0.0
    for data in training_loader:
        left_kmer_profile, right_kmer_profile, labels = data

        # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
        optimizer.zero_grad()
        left_kmer_profile = left_kmer_profile.reshape(
            -1, left_kmer_profile.shape[-1]
        ).to(device)
        right_kmer_profile = right_kmer_profile.reshape(
            -1, right_kmer_profile.shape[-1]
        ).to(device)
        labels = labels.reshape(-1).to(device)

        # Make predictions for the current epoch
        left_output, right_output = model(left_kmer_profile, right_kmer_profile)

        # Compute the loss and backpropagate
        batch_loss = loss_func(left_output, right_output, labels, name=loss_name)
        batch_loss.backward()

        # Update the model parameters
        optimizer.step()

        # Get the epoch loss for reporting
        epoch_loss += batch_loss.item()
        del (
            batch_loss,
            left_kmer_profile,
            right_kmer_profile,
            labels,
            left_output,
            right_output,
        )
        torch.cuda.empty_cache()

    return epoch_loss / len(training_loader)


def run(
    model,
    learning_rate,
    epoch_num,
    training_loader,
    loss_name="bern",
    model_save_path=None,
    checkpoint=0,
    verbose=True,
    wandb_config: dict = None,
):
    # Initialize W&B if config is provided
    if wandb_config is not None and wandb_config.get("enabled", False):
        wandb.init(
            project=wandb_config.get("project", "dna-embedding"),
            entity=wandb_config.get("entity", None),
            mode=wandb_config.get("mode", "online"),
            config=wandb_config.get("config", {}),
            tags=["revisit", "revisit-model"],
        )
        # Log hyperparameters
        if wandb_config.get("config"):
            wandb.config.update(wandb_config["config"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.to(torch.device("cuda:0"))

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        print("Training has just started.")

    for epoch in range(epoch_num):
        if verbose:
            print(f"\t+ Epoch {epoch + 1}.")

        avg_loss = single_epoch(model, loss_func, optimizer, training_loader, loss_name)

        if verbose:
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        # Log to W&B
        if wandb_config is not None and wandb_config.get("enabled", False):
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        if (
            model_save_path is not None
            and checkpoint > 0
            and (epoch + 1) % checkpoint == 0
        ):
            # model_save_path contains the substring epoch=ID, so change the ID to the current epoch
            temp_model_save_path = re.sub(
                "epoch.*_LR", f"epoch={epoch + 1}_LR", model_save_path
            )
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(temp_model_save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            torch.save(
                [
                    {"k": model.get_k(), "device": model.get_device()},
                    model.state_dict(),
                ],
                temp_model_save_path,
            )
            if verbose:
                print(f"Model is saving.")
                print(f"\t- Target path: {temp_model_save_path}")

    if model_save_path is not None:
        # If the model is a DataParallel object, then save the model.module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # Create directory if it doesn't exist
        dir_path = os.path.dirname(model_save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(
            [
                {
                    "k": model.get_k(),
                    "dim": model.get_dim(),
                    "device": model.get_device(),
                },
                model.state_dict(),
            ],
            model_save_path,
        )

        if verbose:
            print(f"Model is saving.")
            print(f"\t- Target path: {model_save_path}")

    # Run evaluation if enabled and model is saved
    if (
        wandb_config is not None
        and wandb_config.get("enabled", False)
        and model_save_path is not None
    ):
        eval_config = wandb_config.get("evaluation", {})
        if eval_config.get("enabled", False):
            try:
                run_evaluation_and_log(
                    model_path=model_save_path,
                    eval_config=eval_config,
                    wandb_config=wandb_config,
                )
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
                if wandb_config.get("enabled", False):
                    wandb.log({"evaluation_error": str(e)})

    # Finish W&B run
    if wandb_config is not None and wandb_config.get("enabled", False):
        wandb.finish()


def run_evaluation_and_log(model_path: str, eval_config: dict, wandb_config: dict):
    """
    Run evaluation using evaluation/binning.py logic and log metrics to W&B.
    Evaluates all available species and samples automatically.
    """
    # Add project root to Python path for evaluation imports
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        print(f"Warning: Could not import evaluation modules: {e}")
        return

    data_dir = eval_config.get("data_dir")
    # Support both single species/sample and multiple
    species_list = eval_config.get("species", "reference")
    if isinstance(species_list, str):
        # If it's a comma-separated string, split it
        if "," in species_list:
            species_list = [s.strip() for s in species_list.split(",")]
        else:
            species_list = [species_list]
    
    sample_list = eval_config.get("sample", [5])
    if isinstance(sample_list, (int, str)):
        # If it's a single value or comma-separated string
        if isinstance(sample_list, str) and "," in sample_list:
            sample_list = [int(s.strip()) for s in sample_list.split(",")]
        else:
            sample_list = [int(sample_list)]
    elif isinstance(sample_list, list):
        sample_list = [int(s) for s in sample_list]
    
    k = eval_config.get("k", 4)
    metric = eval_config.get("metric", "l2")

    if data_dir is None:
        print("Warning: Evaluation data_dir not provided, skipping evaluation")
        return
    
    # Auto-detect available species if data_dir exists
    if os.path.exists(data_dir):
        available_species = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "clustering_0.tsv")
            ):
                available_species.append(item)
        
        if available_species:
            # Use available species if none specified, or filter to available ones
            if not species_list:
                species_list = available_species
                print(f"Auto-detected species: {species_list}")
            else:
                species_list = [s for s in species_list if s in available_species]
                print(f"Evaluating species: {species_list}")

    # Evaluate all species and samples
    all_eval_metrics = {}
    # Collect F1@0.5 and recall@0.5 from all species/samples for aggregation
    f1_at_05_values = []
    recall_at_05_values = []
    
    for species in species_list:
        for sample in sample_list:
            try:
                print(f"\nEvaluating: species={species}, sample={sample}")
                
                # Load clustering data to compute similarity threshold
                clustering_data_file_path = os.path.join(data_dir, species, "clustering_0.tsv")
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
                label2id = {l: i for i, l in enumerate(set(labels))}
                labels = np.array([label2id[l] for l in labels])
                num_clusters = len(label2id)

                # Get embeddings
                embedding = get_embedding(
                    dna_sequences=dna_sequences,
                    model_name="nonlinear",
                    species=species,
                    sample=0,
                    k=k,
                    task_name="clustering",
                    test_model_dir=model_path,
                    suffix="",
                )

                # Compute threshold
                from evaluation.utils import compute_class_center_medium_similarity

                percentile_values = compute_class_center_medium_similarity(
                    embedding, labels, metric=metric
                )
                threshold = percentile_values[-3]

                # Load binning data
                data_file = os.path.join(data_dir, species, f"binning_{sample}.tsv")
                if not os.path.exists(data_file):
                    print(f"Warning: Binning data file not found: {data_file}")
                    continue

                with open(data_file, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                labels_bin = [d[1] for d in data]

                # Filter sequences
                MIN_SEQ_LEN = 2500
                MIN_ABUNDANCE_VALUE = 10
                filterd_idx = [
                    i for i, seq in enumerate(dna_sequences) if len(seq) >= MIN_SEQ_LEN
                ]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                label_counts = collections.Counter(labels_bin)
                filterd_idx = [
                    i
                    for i, l in enumerate(labels_bin)
                    if label_counts[l] >= MIN_ABUNDANCE_VALUE
                ]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                label2id = {l: i for i, l in enumerate(set(labels_bin))}
                labels_bin = np.array([label2id[l] for l in labels_bin])
                num_clusters_bin = len(label2id)

                # Generate embeddings for binning set
                embedding = get_embedding(
                    dna_sequences,
                    "nonlinear",
                    species,
                    sample,
                    k=k,
                    metric=metric,
                    task_name="binning",
                    test_model_dir=model_path,
                    suffix="",
                )

                # Run KMedoid algorithm
                binning_results = KMedoid(
                    embedding,
                    min_similarity=threshold,
                    min_bin_size=10,
                    max_iter=1000,
                    metric=metric,
                    scalable=False,
                )

                # Get metrics
                true_labels_bin = labels_bin[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]

                if len(predicted_labels) == 0:
                    print(f"Warning: No predicted labels after binning for {species} sample {sample}")
                    continue

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(
                    true_labels_bin, predicted_labels
                )
                predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

                # Calculate metrics
                recall_bin = sklearn.metrics.recall_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                recall_bin.sort()

                f1_bin = sklearn.metrics.f1_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                f1_bin.sort()

                thresholds_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                recall_results = []
                f1_results = []
                for thresh in thresholds_list:
                    recall_results.append(len(np.where(recall_bin > thresh)[0]))
                    f1_results.append(len(np.where(f1_bin > thresh)[0]))

                # Store metrics with species and sample prefix
                prefix = f"eval/{species}_sample{sample}"
                all_eval_metrics[f"{prefix}/threshold"] = threshold
                all_eval_metrics[f"{prefix}/num_clusters"] = num_clusters_bin
                all_eval_metrics[f"{prefix}/num_sequences"] = len(dna_sequences)
                all_eval_metrics[f"{prefix}/num_predicted"] = len(predicted_labels)

                for i, thresh in enumerate(thresholds_list):
                    all_eval_metrics[f"{prefix}/recall_at_{thresh}"] = recall_results[i]
                    all_eval_metrics[f"{prefix}/f1_at_{thresh}"] = f1_results[i]
                
                # Collect F1@0.5 and recall@0.5 for aggregation across all species/samples
                f1_at_05_values.append(f1_results[4])  # Index 4 = threshold 0.5
                recall_at_05_values.append(recall_results[4])
                
                print(f"  ✓ Completed: {species} sample {sample} - F1@0.5: {f1_results[4]}")

            except Exception as e:
                print(f"Error evaluating {species} sample {sample}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Compute aggregated summary metrics for sweep optimization
    if f1_at_05_values:
        # Use mean across all species/samples
        all_eval_metrics["eval/f1_at_0.5"] = np.mean(f1_at_05_values)
        all_eval_metrics["eval/recall_at_0.5"] = np.mean(recall_at_05_values)
        all_eval_metrics["eval/f1_at_0.5_std"] = np.std(f1_at_05_values)  # Also log std for insight
        all_eval_metrics["eval/recall_at_0.5_std"] = np.std(recall_at_05_values)
        print(f"\n✓ Aggregated metrics: F1@0.5 = {np.mean(f1_at_05_values):.2f} (std: {np.std(f1_at_05_values):.2f}) across {len(f1_at_05_values)} species/sample combinations")
    
    # Log all metrics to W&B
    if all_eval_metrics:
        wandb.log(all_eval_metrics)
        print(f"\n✓ All evaluation metrics logged to W&B")
        print(f"  Total metrics logged: {len(all_eval_metrics)}")
    else:
        print("\n⚠ No evaluation metrics were generated")

        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering")
    parser.add_argument("--input", type=str, help="Input sequence file")
    parser.add_argument("--k", type=int, default=2, help="k value")
    parser.add_argument("--dim", type=int, default=256, help="dimension value")
    parser.add_argument(
        "--neg_sample_per_pos", type=int, default=1000, help="Negative sample ratio"
    )
    parser.add_argument(
        "--max_read_num",
        type=int,
        default=10000,
        help="Maximum number of reads to get from the file",
    )
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch number")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Batch size (0: no batch)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--workers_num", type=int, default=1, help="Number of workers for data loader"
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        default="bern",
        help="Loss function (bern, poisson, hinge)",
    )
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument(
        "--seed", type=int, default=26042024, help="Seed for random number generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="Save the model for every checkpoint epoch",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (if None, W&B logging is disabled)",
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
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help="Data directory for evaluation (if provided, evaluation will run after training)",
    )
    parser.add_argument(
        "--eval_species",
        type=str,
        default="reference",
        help="Species for evaluation",
    )
    parser.add_argument(
        "--eval_sample",
        type=str,
        default="5",
        help="Sample ID(s) for evaluation (comma-separated, e.g., '5,6')",
    )
    args = parser.parse_args()

    # Handle device selection with fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            print(f"Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device(args.device)

    # Define the model
    model = NonLinearModel(
        k=args.k,
        dim=args.dim,
        device=device,
        verbose=True,
        seed=args.seed,
    )

    # Read the dataset
    training_dataset = PairDataset(
        file_path=args.input,
        transform_func=model.read2kmer_profile,
        neg_sample_per_pos=args.neg_sample_per_pos,
        max_read_num=args.max_read_num,
        seed=args.seed,
    )
    # Define the training data loader
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size if args.batch_size else len(training_dataset),
        shuffle=True,
        num_workers=args.workers_num,
    )

    # Prepare W&B config
    wandb_config = None
    if args.wandb_project is not None and args.wandb_mode != "disabled":
        wandb_config = {
            "enabled": True,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
            "config": {
                "k": args.k,
                "dim": args.dim,
                "lr": args.lr,
                "epoch": args.epoch,
                "batch_size": args.batch_size
                if args.batch_size
                else len(training_dataset),
                "neg_sample_per_pos": args.neg_sample_per_pos,
                "loss_name": args.loss_name,
                "max_read_num": args.max_read_num,
                "seed": args.seed,
                "device": args.device,
                "workers_num": args.workers_num,
            },
            "evaluation": {
                "enabled": args.eval_data_dir is not None,
                "data_dir": args.eval_data_dir,
                "species": args.eval_species,
                "sample": args.eval_sample,
                "k": args.k,
                "metric": "l2",
            },
        }

    # Run the model
    run(
        model,
        args.lr,
        args.epoch,
        training_loader,
        args.loss_name,
        args.output,
        args.checkpoint,
        verbose=True,
        wandb_config=wandb_config,
    )
