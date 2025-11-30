import argparse
import itertools
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import os


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


def run(
    model,
    learning_rate: float,
    epoch_num: int,
    training_loader,
    model_save_path: str = None,
    checkpoint: int = 0,
    verbose: bool = True,
    temperature: float = 0.1,
    wandb_config: dict = None,
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

    for epoch in range(epoch_num):
        if verbose:
            print(f"\t+ Epoch {epoch + 1}.")

        avg_loss = single_epoch(
            model, optimizer, training_loader, temperature=temperature
        )

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
            temp_model_save_path = re.sub(
                "epoch.*_LR", f"epoch={epoch + 1}_LR", model_save_path
            )
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
    Run evaluation using evaluation/binning.py logic and log metrics to W&B.
    This is a simplified version that can be extended based on evaluation needs.
    """
    # Import evaluation utilities
    try:
        from evaluation.utils import get_embedding
        from evaluation.binning import KMedoid, align_labels_via_hungarian_algorithm
        import sklearn.metrics
        import collections
        import csv
    except ImportError as e:
        print(f"Warning: Could not import evaluation modules: {e}")
        return

    data_dir = eval_config.get("data_dir")
    species = eval_config.get("species", "reference")
    sample = eval_config.get("sample", 5)
    k = eval_config.get("k", 4)
    metric = eval_config.get("metric", "l2")

    if data_dir is None:
        print("Warning: Evaluation data_dir not provided, skipping evaluation")
        return

    try:
        # Load clustering data to compute similarity threshold
        clustering_data_file_path = os.path.join(data_dir, species, "clustering_0.tsv")
        if not os.path.exists(clustering_data_file_path):
            print(
                f"Warning: Clustering data file not found: {clustering_data_file_path}"
            )
            return

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
            return

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
            print("Warning: No predicted labels after binning")
            return

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

        # Log to W&B
        eval_metrics = {
            "eval/threshold": threshold,
            "eval/num_clusters": num_clusters_bin,
            "eval/num_sequences": len(dna_sequences),
            "eval/num_predicted": len(predicted_labels),
        }

        for i, thresh in enumerate(thresholds_list):
            eval_metrics[f"eval/recall_at_{thresh}"] = recall_results[i]
            eval_metrics[f"eval/f1_at_{thresh}"] = f1_results[i]

        wandb.log(eval_metrics)
        print(f"Evaluation metrics logged to W&B: {eval_metrics}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SupCon multi-view genome representation"
    )
    parser.add_argument(
        "--input", type=str, help="Input sequence file (left,right per line)"
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
        type=int,
        default=5,
        help="Sample ID for evaluation",
    )
    args = parser.parse_args()

    set_seed(args.seed)

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
                "species": args.eval_species,
                "sample": args.eval_sample,
                "k": args.k,
                "metric": "l2",
            },
        }

    run(
        model,
        args.lr,
        args.epoch,
        training_loader,
        model_save_path=args.output,
        checkpoint=args.checkpoint,
        verbose=True,
        temperature=args.temperature,
        wandb_config=wandb_config,
    )
