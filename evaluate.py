"""Emergence evaluation entry point.

Computes:
  1. MLM accuracy + perplexity on human proteins (trained model vs random init)
  2. UMAP/t-SNE representation analysis
  3. Attention entropy per layer
  4. Emergence score
  5. Saves results to outputs/figures/

Usage:
    python evaluate.py
    python evaluate.py --checkpoint outputs/checkpoints/ckpt_best.pt
    python evaluate.py --no-plots
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.config_loader import load_config
from src.dataset import ProteinDataset
from src.model import build_model, ProteinLM
from src.tokenizer import ProteinTokenizer
from src.emergence import (
    compute_mlm_metrics,
    extract_representations,
    reduce_representations,
    compute_emergence_score,
    compute_attention_entropy,
    plot_representations,
    plot_emergence_summary,
    aa_freq_kl_divergence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no-plots",   action="store_true")
    return p.parse_args()


def get_device(cfg: dict) -> torch.device:
    hw = cfg["hardware"]
    if hw["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(hw["device"])


def find_best_checkpoint(save_dir: Path) -> Path | None:
    candidates = list(save_dir.glob("ckpt_*best*.pt"))
    if not candidates:
        candidates = sorted(save_dir.glob("ckpt_*.pt"))
    return candidates[-1] if candidates else None


def load_trained_model(ckpt_path: Path, cfg: dict, vocab_size: int, device: torch.device) -> ProteinLM:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg["model"], vocab_size)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def make_random_model(cfg: dict, vocab_size: int, device: torch.device) -> ProteinLM:
    model = build_model(cfg["model"], vocab_size)
    model = model.to(device)
    logger.info("Created random-init baseline model")
    return model


def make_dataloader(fasta_paths, tokenizer, cfg, mode="eval") -> DataLoader:
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ds = ProteinDataset(
        fasta_paths = fasta_paths,
        tokenizer   = tokenizer,
        max_len     = dcfg["max_seq_len"],
        mode        = mode,
    )
    return DataLoader(
        ds,
        batch_size  = tcfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = dcfg["num_workers"],
        pin_memory  = dcfg["pin_memory"] and torch.cuda.is_available(),
    )


# -----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg  = load_config(args.config)
    dcfg = cfg["data"]
    ecfg = cfg["evaluation"]

    device = get_device(cfg)
    logger.info(f"Device: {device}")

    tokenizer = ProteinTokenizer(
        mask_prob        = cfg["tokenizer"]["mask_prob"],
        mask_token_prob  = cfg["tokenizer"]["mask_token_prob"],
        random_token_prob= cfg["tokenizer"]["random_token_prob"],
    )

    proc_dir = Path(dcfg["processed_dir"])

    # ---- Data loaders ----
    train_files = [proc_dir / f"{t['name'].lower()}_{t['id']}.fasta" for t in dcfg["train_taxa"]]
    human_files = [proc_dir / f"{t['name'].lower()}_{t['id']}.fasta" for t in dcfg["eval_taxa"]]

    missing = [f for f in train_files + human_files if not f.exists()]
    if missing:
        logger.error(f"Missing files: {missing}\nRun: python data/download_data.py")
        sys.exit(1)

    train_dl = make_dataloader(train_files, tokenizer, cfg, mode="train")
    human_dl = make_dataloader(human_files, tokenizer, cfg, mode="train")

    # ---- Models ----
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = ecfg.get("checkpoint") or find_best_checkpoint(Path(cfg["training"]["save_dir"]))
    if ckpt_path is None:
        logger.error("No checkpoint found. Train the model first: python train.py")
        sys.exit(1)

    trained_model = load_trained_model(Path(ckpt_path), cfg, tokenizer.vocab_size, device)
    random_model  = make_random_model(cfg, tokenizer.vocab_size, device)

    prec = cfg["training"]["precision"]
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]

    out_dir = Path(ecfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ----------------------------------------------------------------
    # 1. MLM metrics on training domain
    # ----------------------------------------------------------------
    logger.info("=== Computing metrics on TRAINING domain (bacteria/fungi) ===")
    trained_train_metrics = compute_mlm_metrics(trained_model, train_dl, device, dtype)
    logger.info(f"  Trained model on train: {trained_train_metrics}")
    results["trained_on_train"] = trained_train_metrics

    # ----------------------------------------------------------------
    # 2. MLM metrics on human proteins (TRAINED model)
    # ----------------------------------------------------------------
    logger.info("=== Computing metrics on HUMAN proteins (trained model) ===")
    trained_human_metrics = compute_mlm_metrics(trained_model, human_dl, device, dtype)
    logger.info(f"  Trained model on human: {trained_human_metrics}")
    results["trained_on_human"] = trained_human_metrics

    # ----------------------------------------------------------------
    # 3. MLM metrics on human proteins (RANDOM model — lower bound)
    # ----------------------------------------------------------------
    logger.info("=== Computing metrics on HUMAN proteins (random init — baseline) ===")
    random_human_metrics = compute_mlm_metrics(random_model, human_dl, device, dtype)
    logger.info(f"  Random model on human: {random_human_metrics}")
    results["random_on_human"] = random_human_metrics

    # ----------------------------------------------------------------
    # 4. Emergence score
    # ----------------------------------------------------------------
    logger.info("=== Computing emergence score ===")
    emergence = compute_emergence_score(
        trained_metrics = trained_train_metrics,
        random_metrics  = random_human_metrics,
        human_metrics   = trained_human_metrics,
    )
    logger.info(f"  Emergence scores: {json.dumps(emergence, indent=2)}")
    results["emergence_scores"] = emergence

    # ----------------------------------------------------------------
    # 5. Attention entropy
    # ----------------------------------------------------------------
    logger.info("=== Attention entropy analysis ===")
    trained_attn_entropy = compute_attention_entropy(trained_model, human_dl, device)
    random_attn_entropy  = compute_attention_entropy(random_model, human_dl, device)
    results["attention_entropy_trained"] = trained_attn_entropy
    results["attention_entropy_random"]  = random_attn_entropy

    # ----------------------------------------------------------------
    # 6. KL divergence on amino-acid distribution
    # ----------------------------------------------------------------
    logger.info("=== AA frequency KL divergence ===")
    kl_trained = aa_freq_kl_divergence(trained_model, human_dl, device, tokenizer.vocab_size)
    kl_random  = aa_freq_kl_divergence(random_model,  human_dl, device, tokenizer.vocab_size)
    results["aa_kl_trained_on_human"] = kl_trained
    results["aa_kl_random_on_human"]  = kl_random
    logger.info(f"  KL divergence — trained: {kl_trained:.4f}  random: {kl_random:.4f}")

    # ----------------------------------------------------------------
    # 7. Save all results
    # ----------------------------------------------------------------
    result_path = out_dir / "emergence_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {result_path}")

    # ----------------------------------------------------------------
    # 8. Representation visualisation (optional)
    # ----------------------------------------------------------------
    if not args.no_plots:
        logger.info("=== Extracting representations for visualisation ===")
        repr_cfg = ecfg["emergence"]["repr_analysis"]
        n_samples = repr_cfg["n_samples"]
        method    = repr_cfg["method"]
        layer_idx = repr_cfg["layer_indices"][0]

        reps_train_trained = extract_representations(trained_model, train_dl, device, layer_idx, n_samples // 2)
        reps_human_trained = extract_representations(trained_model, human_dl, device, layer_idx, n_samples // 2)
        reps_human_random  = extract_representations(random_model,  human_dl, device, layer_idx, n_samples // 2)

        combined = np.concatenate([reps_train_trained, reps_human_trained, reps_human_random])
        reduced  = reduce_representations(combined, method=method)

        n1, n2, n3 = len(reps_train_trained), len(reps_human_trained), len(reps_human_random)
        emb_dict = {
            "Train (bacteria/fungi)": reduced[:n1],
            "Human (trained model)":  reduced[n1:n1+n2],
            "Human (random model)":   reduced[n1+n2:],
        }
        plot_representations(
            emb_dict,
            output_path = out_dir / "representations.png",
            title       = f"Protein Representations ({method.upper()}) — Trained Model",
        )
        plot_emergence_summary(emergence, out_dir / "emergence_summary.png")

    logger.info("Evaluation complete.")
    _print_report(results)


def _print_report(results: dict):
    em = results.get("emergence_scores", {})
    logger.info("\n" + "=" * 60)
    logger.info("  EMERGENCE ANALYSIS REPORT")
    logger.info("=" * 60)
    logger.info(f"  Trained model | Human accuracy:  {em.get('trained_human_accuracy', 0):.4f}")
    logger.info(f"  Random model  | Human accuracy:  {em.get('random_human_accuracy', 0):.4f}")
    logger.info(f"  Trained model | Train accuracy:  {em.get('trained_train_accuracy', 0):.4f}")
    logger.info(f"  Transfer ratio (human/train):    {em.get('transfer_ratio', 0):.4f}")
    logger.info(f"  Accuracy improvement over random:{em.get('emergence_acc_improvement', 0):.4f}")
    logger.info(f"  CE improvement over random:      {em.get('emergence_ce_improvement', 0):.4f}")
    logger.info("=" * 60)
    if em.get("emergence_acc_improvement", 0) > 0.1:
        logger.info("  >> Strong emergence signal detected! <<")
    elif em.get("emergence_acc_improvement", 0) > 0:
        logger.info("  >> Mild emergence signal detected.")
    else:
        logger.info("  >> No clear emergence signal at this scale.")


if __name__ == "__main__":
    main()
