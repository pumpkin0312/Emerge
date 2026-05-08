"""Emergence analysis: compare model performance on human vs. training-domain proteins."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------
@torch.no_grad()
def compute_mlm_metrics(
    model,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype=None,
) -> dict[str, float]:
    """Return MLM accuracy and perplexity on the given dataloader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype else _nullcontext()
        )
        with ctx:
            out  = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                out["logits"].view(-1, out["logits"].size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        total_loss += loss.item()
        mask   = labels != -100
        preds  = out["logits"].argmax(-1)
        correct += (preds[mask] == labels[mask]).sum().item()
        total   += mask.sum().item()

    n        = len(dataloader)
    avg_loss = total_loss / max(n, 1)
    return {
        "mlm_accuracy": correct / max(total, 1),
        "perplexity":   math.exp(avg_loss),
        "cross_entropy": avg_loss,
        "n_masked_tokens": total,
    }


@torch.no_grad()
def extract_representations(
    model,
    dataloader: DataLoader,
    device: torch.device,
    layer_index: int = -1,
    max_samples: int = 2000,
) -> np.ndarray:
    """Extract mean-pooled sequence representations for dimensionality reduction."""
    model.eval()
    reps = []
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        rep = model.get_representations(input_ids, attention_mask, layer_index)
        reps.append(rep.cpu().float().numpy())
        if sum(r.shape[0] for r in reps) >= max_samples:
            break
    arr = np.concatenate(reps, axis=0)
    return arr[:max_samples]


# -----------------------------------------------------------------------
# Dimensionality reduction
# -----------------------------------------------------------------------
def reduce_representations(
    reps: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
) -> np.ndarray:
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(reps) - 1))

    return reducer.fit_transform(reps)


# -----------------------------------------------------------------------
# Amino-acid frequency divergence
# -----------------------------------------------------------------------
def aa_freq_kl_divergence(
    model,
    dataloader: DataLoader,
    device: torch.device,
    vocab_size: int,
) -> float:
    """
    KL divergence between predicted token distribution and true distribution
    on masked positions — measures distributional alignment with target domain.
    """
    model.eval()
    pred_counts = torch.zeros(vocab_size)
    true_counts = torch.zeros(vocab_size)

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            mask   = labels != -100
            if not mask.any():
                continue

            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = F.softmax(out["logits"], dim=-1).cpu()

            pred_counts += probs[mask].sum(0)
            for lbl in labels[mask]:
                true_counts[lbl.item()] += 1

    pred_dist = pred_counts / pred_counts.sum().clamp(min=1e-9)
    true_dist = true_counts / true_counts.sum().clamp(min=1e-9)

    kl = (true_dist * (true_dist / (pred_dist + 1e-9) + 1e-9).log()).sum().item()
    return float(kl)


# -----------------------------------------------------------------------
# Emergence score
# -----------------------------------------------------------------------
def compute_emergence_score(
    trained_metrics:     dict[str, float],
    random_metrics:      dict[str, float],
    human_metrics:       dict[str, float],
) -> dict[str, float]:
    """
    Emergence score:
      E = (trained_human - random_human) / (random_human - trained_train + ε)

    E >> 0 implies the model acquired knowledge that generalises beyond its
    training domain — a candidate emergence signal.
    """
    eps = 1e-6

    # Use cross-entropy as primary metric (lower = better)
    ce_trained_human = human_metrics["cross_entropy"]
    ce_random_human  = random_metrics["cross_entropy"]

    acc_trained_human = human_metrics["mlm_accuracy"]
    acc_random_human  = random_metrics["mlm_accuracy"]
    acc_trained_train = trained_metrics["mlm_accuracy"]

    # Normalised improvement over random baseline
    improvement_ce  = (ce_random_human  - ce_trained_human)  / (ce_random_human  + eps)
    improvement_acc = (acc_trained_human - acc_random_human) / (acc_random_human + eps)

    # Transfer ratio: how much of domain accuracy carries over to human
    transfer_ratio = acc_trained_human / (acc_trained_train + eps)

    return {
        "emergence_ce_improvement":   improvement_ce,
        "emergence_acc_improvement":  improvement_acc,
        "transfer_ratio":             transfer_ratio,
        "trained_human_accuracy":     acc_trained_human,
        "random_human_accuracy":      acc_random_human,
        "trained_train_accuracy":     acc_trained_train,
        "trained_human_perplexity":   human_metrics["perplexity"],
        "random_human_perplexity":    random_metrics["perplexity"],
    }


# -----------------------------------------------------------------------
# Attention analysis
# -----------------------------------------------------------------------
@torch.no_grad()
def compute_attention_entropy(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, float]:
    """Average attention entropy per layer — proxy for how 'focused' the model is."""
    model.eval()
    layer_entropies: list[list[float]] = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        out = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            output_attentions=True,
        )
        attns = out["attentions"]  # (B, num_layers, num_heads, L, L)
        for layer_idx in range(attns.shape[1]):
            attn_layer = attns[:, layer_idx, :, :, :]  # (B, H, L, L)
            # entropy over last dim
            attn_layer = attn_layer.clamp(min=1e-9)
            entropy = -(attn_layer * attn_layer.log()).sum(-1).mean().item()
            if len(layer_entropies) <= layer_idx:
                layer_entropies.append([])
            layer_entropies[layer_idx].append(entropy)

    return {
        f"layer_{i}_attn_entropy": float(np.mean(vals))
        for i, vals in enumerate(layer_entropies)
    }


# -----------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------
def plot_representations(
    embeddings_dict: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Protein Representations",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for (label, emb), color in zip(embeddings_dict.items(), colors):
        ax.scatter(emb[:, 0], emb[:, 1], label=label, alpha=0.5, s=10, c=color)

    ax.set_title(title)
    ax.legend(markerscale=2)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved representation plot → {output_path}")


def plot_emergence_summary(
    scores: dict[str, float],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    keys   = [k for k in scores if "accuracy" in k or "perplexity" in k]
    values = [scores[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(keys, values, color="#4C72B0")
    ax.set_xlabel("Value")
    ax.set_title("Emergence Analysis Summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved emergence summary → {output_path}")


# -----------------------------------------------------------------------
class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *_): pass
