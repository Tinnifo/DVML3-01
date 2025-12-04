import argparse
import itertools
import random
import re
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import os

# Suppress urllib3 OpenSSL warning on macOS
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
except (ImportError, AttributeError):
    pass


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def reverse_complement(seq: str) -> str:
    comp_map = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp_map)[::-1]



def split_read(seq: str, k: int, L_min_useful: int, W_target: int):
    """
    Length-aware splitting:
      - if len < k:  no valid k-mers -> []
      - if k <= len < L_min_useful:  no split, just [seq]
      - if L_min_useful <= len < W_target: up to 2 overlapping windows
      - if len >= W_target: sliding windows of size W_target with 50% overlap
    """
    L = len(seq)
    if L < k:
        return []

    if L < L_min_useful:
        return [seq]

    if L < W_target:
        window_size = max(L // 2, L_min_useful)
        if window_size >= L:
            return [seq]
        v1 = seq[0:window_size]
        v2 = seq[-window_size:]
        views = [v1]
        if v2 != v1:
            views.append(v2)
        return views

    window_size = W_target
    stride = window_size // 2
    views = []
    start = 0
    while start + window_size <= L:
        views.append(seq[start : start + window_size])
        start += stride

    tail_len = L - start
    if tail_len >= L_min_useful:
        last = seq[-window_size:]
        if not views or last != views[-1]:
            views.append(last)

    return views




def make_views_for_read(
    seq: str,
    k: int,
    max_views_per_read: int,
    L_min_useful: int,
    W_target: int,
    use_reverse_complement: bool = True,
):
    """
    Returns a list of sequence views (strings) for one read.
    Ensures at least 1 view if len(seq) >= k.
    """
    base_views = split_read(seq, k, L_min_useful=L_min_useful, W_target=W_target)
    views = []

    for v in base_views:
        views.append(v)
        if use_reverse_complement:
            views.append(reverse_complement(v))
        if len(views) >= max_views_per_read:
            break

    if not views and len(seq) >= k:
        # fallback: use the whole read (and maybe RC) if splitting returned nothing
        views = [seq]
        if use_reverse_complement and max_views_per_read > 1:
            rc = reverse_complement(seq)
            if rc != seq:
                views.append(rc)

    return views[:max_views_per_read]



class SupConPairDataset(Dataset):
    """
    Multi-view SupCon dataset.

    Each line in file: left_read,right_read

    __getitem__(idx) returns:
      - kmer_profiles: [n_views, D]
      - frag_ids:      [n_views]  (all = idx)
    """

    def __init__(
        self,
        file_path: str,
        transform_func,
        k: int,
        max_read_num: int = 0,
        max_views_per_read: int = 4,
        L_min_useful: int = 64,
        W_target: int = 256,
        use_reverse_complement: bool = True,
        verbose: bool = True,
        seed: int = 0,
    ):
        self.__transform_func = transform_func
        self.k = k
        self.max_views_per_read = max_views_per_read
        self.L_min_useful = L_min_useful
        self.W_target = W_target
        self.use_reverse_complement = use_reverse_complement

        set_seed(seed)

        # Count lines
        with open(file_path, "r") as f:
            lines_num = sum(1 for _ in f)

        # Optional subsampling
        if max_read_num > 0:
            sample_size = min(max_read_num, lines_num)
            chosen_lines = random.sample(range(lines_num), sample_size)
            chosen_lines.sort()
        else:
            chosen_lines = None

        chosen_line_idx = 0
        self.left_reads = []
        self.right_reads = []

        # Read raw sequences
        with open(file_path, "r") as f:
            for current_line_idx, line in enumerate(f):
                if chosen_lines is not None:
                    if chosen_line_idx == len(chosen_lines):
                        break
                    if current_line_idx != chosen_lines[chosen_line_idx]:
                        continue
                    else:
                        chosen_line_idx += 1

                left_read, right_read = line.strip().split(",")
                self.left_reads.append(left_read)
                self.right_reads.append(right_read)

        if verbose:
            print("The data file was read successfully!")
            print(f"\t+ Total number of read pairs in file: {lines_num}")
            print(f"\t+ Number of read pairs used: {len(self.left_reads)}")

    def __len__(self):
        return len(self.left_reads)

    def __getitem__(self, idx):
        left_seq = self.left_reads[idx]
        right_seq = self.right_reads[idx]

        # Generate views for each read
        left_views = make_views_for_read(
            left_seq,
            k=self.k,
            max_views_per_read=self.max_views_per_read,
            L_min_useful=self.L_min_useful,
            W_target=self.W_target,
            use_reverse_complement=self.use_reverse_complement,
        )
        right_views = make_views_for_read(
            right_seq,
            k=self.k,
            max_views_per_read=self.max_views_per_read,
            L_min_useful=self.L_min_useful,
            W_target=self.W_target,
            use_reverse_complement=self.use_reverse_complement,
        )

        all_seqs = left_views + right_views  # n_views sequences

        # Convert each view to k-mer profile
        kmer_profiles = [self.__transform_func(s) for s in all_seqs]
        kmer_profiles = torch.from_numpy(np.asarray(kmer_profiles)).to(
            torch.float
        )  # [n_views, D]

        # Same fragment id for all views of this pair
        frag_ids = torch.full(
            (kmer_profiles.shape[0],), idx, dtype=torch.long
        )  # [n_views]

        return kmer_profiles, frag_ids


def supcon_collate_fn(batch):
    """
    batch: list of (kmer_profiles_i, frag_ids_i), where:
      - kmer_profiles_i: [n_i, D]
      - frag_ids_i:      [n_i]
    Returns:
      - kmer_profiles: [sum_i n_i, D]
      - frag_ids:      [sum_i n_i]
    """
    kmer_list = []
    id_list = []
    for kmers, ids in batch:
        kmer_list.append(kmers)
        id_list.append(ids)
    kmer_profiles = torch.cat(kmer_list, dim=0)
    frag_ids = torch.cat(id_list, dim=0)
    return kmer_profiles, frag_ids


class NonLinearModel(torch.nn.Module):
    def __init__(
        self,
        k: int,
        dim: int = 256,
        device=torch.device("cpu"),
        verbose: bool = False,
        seed: int = 0,
    ):
        super(NonLinearModel, self).__init__()

        self.__device = device
        self.__verbose = verbose
        self.__k = k
        self.__dim = dim
        self.__letters = ["A", "C", "G", "T"]
        self.__kmer2id = {
            "".join(kmer): i
            for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))
        }
        self.__kmers_num = len(self.__kmer2id)

        set_seed(seed)

        self.linear1 = torch.nn.Linear(
            self.__kmers_num, 512, dtype=torch.float, device=self.__device
        )
        self.batch1 = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self.__device)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(
            512, self.__dim, dtype=torch.float, device=self.__device
        )

        if self.__verbose:
            print(f"NonLinearModel initialized with k={k}, dim={dim}, device={device}")
            print(f"Number of k-mers: {self.__kmers_num}")

    def encoder(self, kmer_profile: torch.Tensor) -> torch.Tensor:
        output = self.linear1(kmer_profile)
        output = self.batch1(output)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        return output

    def get_k(self):
        return self.__k

    def get_dim(self):
        return self.__dim

    def get_device(self):
        return self.__device

    def read2kmer_profile(self, read: str, normalized: bool = True):
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


def supcon_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
):
    """
    Supervised Contrastive Loss with configurable temperature.
    """
    """
    embeddings: [N, d]  (all views in the batch)
    labels:     [N]     (fragment IDs; same ID = positives)
    """
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    N = z.size(0)

    sim = torch.matmul(z, z.T) / temperature
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, -1e9)

    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T) & ~self_mask

    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0

    loss = torch.zeros_like(pos_counts, dtype=torch.float, device=device)
    loss[valid] = -(log_prob * pos_mask).sum(dim=1)[valid] / (pos_counts[valid] + 1e-12)
    return loss[valid].mean()


def single_epoch(model, optimizer, training_loader, temperature: float = 0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0.0

    for kmer_profiles, frag_ids in training_loader:
        optimizer.zero_grad()
        kmer_profiles = kmer_profiles.to(device)
        frag_ids = frag_ids.to(device)

        embeddings = model.encoder(kmer_profiles)
        batch_loss = supcon_loss(embeddings, frag_ids, temperature=temperature)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        del batch_loss
        torch.cuda.empty_cache()

    return epoch_loss / len(training_loader)


def validate_epoch(model, validation_loader, temperature: float = 0.1):
    """
    Validate the model on validation data without updating weights.
    Returns validation loss.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for kmer_profiles, frag_ids in validation_loader:
            kmer_profiles = kmer_profiles.to(device)
            frag_ids = frag_ids.to(device)

            embeddings = model.encoder(kmer_profiles)
            batch_loss = supcon_loss(embeddings, frag_ids, temperature=temperature)

            val_loss += batch_loss.item()
            del batch_loss
            torch.cuda.empty_cache()

    model.train()
    return val_loss / len(validation_loader)


def run(
    model,
    learning_rate: float,
    epoch_num: int,
    training_loader,
    validation_loader=None,
    model_save_path: str = None,
    checkpoint: int = 0,
    verbose: bool = True,
    temperature: float = 0.1,
    wandb_config: dict = None,
    early_stopping_patience: int = 0,
):
    # Initialize W&B if config is provided
    if wandb_config is not None and wandb_config.get("enabled", False):
        wandb.init(
            project=wandb_config.get("project", "dna-embedding"),
            entity=wandb_config.get("entity", None),
            mode=wandb_config.get("mode", "online"),
            config=wandb_config.get("config", {}),
            tags=["our", "hour-model"],
        )
        # Log hyperparameters
        if wandb_config.get("config"):
            wandb.config.update(wandb_config["config"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.to(torch.device("cuda:0"))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        print("Training has just started.")
        if validation_loader is not None:
            print("Validation enabled - will compute validation loss each epoch.")

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epoch_num):
        if verbose:
            print(f"\t+ Epoch {epoch + 1}.")

        avg_loss = single_epoch(
            model, optimizer, training_loader, temperature=temperature
        )

        # Compute validation loss if validation loader is provided
        val_loss = None
        if validation_loader is not None:
            val_loss = validate_epoch(model, validation_loader, temperature=temperature)
            if verbose:
                print(
                    f"Epoch {epoch + 1}, Training Loss: {avg_loss}, Validation Loss: {val_loss}"
                )

            # Early stopping logic
            if early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    if isinstance(model, torch.nn.DataParallel):
                        best_model_state = model.module.state_dict().copy()
                    else:
                        best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(
                                f"Early stopping triggered after {epoch + 1} epochs (patience: {early_stopping_patience})"
                            )
                        # Restore best model
                        if best_model_state is not None:
                            if isinstance(model, torch.nn.DataParallel):
                                model.module.load_state_dict(best_model_state)
                            else:
                                model.load_state_dict(best_model_state)
                        break
        else:
            if verbose:
                print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        # Log to W&B
        if wandb_config is not None and wandb_config.get("enabled", False):
            log_dict = {"train_loss": avg_loss, "epoch": epoch + 1}
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict)

        if (
            model_save_path is not None
            and checkpoint > 0
            and (epoch + 1) % checkpoint == 0
        ):
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
                print("Model is saving.")
                print(f"\t- Target path: {temp_model_save_path}")

    if model_save_path is not None:
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
            print("Model is saving.")
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
    Run evaluation using evaluation and log metrics to W&B.
    """
    # Add project root to Python path for evaluation imports
    import sys
    import traceback

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
                label2id = {l: i for i, l in enumerate(set(labels))}
                labels = np.array([label2id[l] for l in labels])
                num_clusters = len(label2id)

                # Get embeddings
                embedding = get_embedding(
                    dna_sequences=dna_sequences,
                    model_name="our",
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
                    "our",
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
                    print(
                        f"Warning: No predicted labels after binning for {species} sample {sample}"
                    )
                    continue

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(
                    true_labels_bin, predicted_labels
                )
                predicted_labels_bin = [
                    alignment_bin[label] for label in predicted_labels
                ]

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

                print(
                    f"  ✓ Completed: {species} sample {sample} - F1@0.5: {f1_results[4]}"
                )

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
        all_eval_metrics["eval/f1_at_0.5_std"] = np.std(
            f1_at_05_values
        )  # Also log std for insight
        all_eval_metrics["eval/recall_at_0.5_std"] = np.std(recall_at_05_values)
        print(
            f"\n✓ Aggregated metrics: F1@0.5 = {np.mean(f1_at_05_values):.2f} (std: {np.std(f1_at_05_values):.2f}) across {len(f1_at_05_values)} species/sample combinations"
        )

    # Log all metrics to W&B
    if all_eval_metrics:
        wandb.log(all_eval_metrics)
        # Also log as summary for better visibility in W&B UI
        for key, value in all_eval_metrics.items():
            wandb.run.summary[key] = value
        wandb.run.summary.update(all_eval_metrics)
        print(f"\n✓ All evaluation metrics logged to W&B")
        print(f"  Total metrics logged: {len(all_eval_metrics)}")
    else:
        print("\n⚠ No evaluation metrics were generated")
        traceback.print_exc()
        # Don't raise - just log the error so the run completes
        wandb.log({"evaluation_warning": "No evaluation metrics were generated"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SupCon multi-view genome representation"
    )
    parser.add_argument(
        "--input", type=str, help="Input sequence file (left,right per line)"
    )
    parser.add_argument(
        "--val_input",
        type=str,
        default=None,
        help="Validation sequence file (left,right per line). If provided, validation loss will be computed each epoch.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Early stopping patience (number of epochs without improvement). 0 = disabled.",
    )
    parser.add_argument("--k", type=int, default=4, help="k-mer size")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension")
    parser.add_argument(
        "--max_read_num",
        type=int,
        default=10000,
        help="Maximum number of read pairs to get from the file",
    )
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch number")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Batch size (0: use full dataset)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--workers_num", type=int, default=1, help="Number of workers for data loader"
    )
    parser.add_argument(
        "--output", type=str, help="Output model path (prefix or filename)"
    )
    parser.add_argument(
        "--seed", type=int, default=26042024, help="Seed for random number generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="Save the model every N epochs (0: only at the end)",
    )
    parser.add_argument(
        "--max_views_per_read",
        type=int,
        default=4,
        help="Maximum number of views generated per read",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter for SupCon loss",
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

    set_seed(args.seed)

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
    model = NonLinearModel(
        k=args.k,
        dim=args.dim,
        device=device,
        verbose=True,
        seed=args.seed,
    )

    # Heuristics for view splitting
    L_min_useful = 4 * (args.k**2)  # e.g. k=4 -> 64
    W_target = 4 * L_min_useful  # e.g. -> 256

    training_dataset = SupConPairDataset(
        file_path=args.input,
        transform_func=model.read2kmer_profile,
        k=args.k,
        max_read_num=args.max_read_num,
        max_views_per_read=args.max_views_per_read,
        L_min_useful=L_min_useful,
        W_target=W_target,
        use_reverse_complement=True,
        seed=args.seed,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size if args.batch_size else len(training_dataset),
        shuffle=True,
        num_workers=args.workers_num,
        collate_fn=supcon_collate_fn,
    )

    # Create validation dataset and loader if validation file is provided
    validation_loader = None
    if args.val_input is not None:
        validation_dataset = SupConPairDataset(
            file_path=args.val_input,
            transform_func=model.read2kmer_profile,
            k=args.k,
            max_read_num=0,  # Use all validation data
            max_views_per_read=args.max_views_per_read,
            L_min_useful=L_min_useful,
            W_target=W_target,
            use_reverse_complement=True,
            seed=args.seed,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size if args.batch_size else len(validation_dataset),
            shuffle=False,  # Don't shuffle validation data
            num_workers=args.workers_num,
            collate_fn=supcon_collate_fn,
        )
        print(f"Validation dataset loaded: {len(validation_dataset)} samples")

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
                "max_views_per_read": args.max_views_per_read,
                "max_read_num": args.max_read_num,
                "seed": args.seed,
                "temperature": args.temperature,
                "device": args.device,
                "workers_num": args.workers_num,
            },
            "evaluation": {
                "enabled": args.eval_data_dir is not None,
                "data_dir": args.eval_data_dir,
                "species": args.eval_species,  # Can be comma-separated string or list
                "sample": args.eval_sample,  # Can be comma-separated string or list
                "k": args.k,
                "metric": "l2",
            },
        }

    run(
        model,
        args.lr,
        args.epoch,
        training_loader,
        validation_loader=validation_loader,
        model_save_path=args.output,
        checkpoint=args.checkpoint,
        verbose=True,
        temperature=args.temperature,
        wandb_config=wandb_config,
        early_stopping_patience=args.early_stopping_patience,
    )
