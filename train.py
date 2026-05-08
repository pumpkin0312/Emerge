"""Main training entry point.

Usage:
    python train.py
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --preset medium
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.config_loader import load_config, save_config
from src.dataset import ProteinDataset, split_dataset
from src.model import build_model
from src.tokenizer import ProteinTokenizer
from src.trainer import Trainer

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/logs/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--preset", default=None, help="Override model preset: tiny/small/medium/large")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    hw = cfg["hardware"]
    if hw["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(hw["device"])


# -----------------------------------------------------------------------
def build_dataloaders(cfg: dict, tokenizer: ProteinTokenizer):
    dcfg  = cfg["data"]
    tcfg  = cfg["training"]
    proc  = Path(dcfg["processed_dir"])

    train_taxa = dcfg["train_taxa"]
    train_files = [proc / f"{t['name'].lower()}_{t['id']}.fasta" for t in train_taxa]

    # Check files exist
    missing = [f for f in train_files if not f.exists()]
    if missing:
        logger.error(f"Missing data files: {missing}")
        logger.error("Run: python data/download_data.py")
        sys.exit(1)

    train_ds = ProteinDataset(
        fasta_paths = train_files,
        tokenizer   = tokenizer,
        max_len     = dcfg["max_seq_len"],
        mode        = "train",
    )

    # split train/val
    n_val  = max(1, int(len(train_ds) * dcfg["val_ratio"]))
    n_train = len(train_ds) - n_val
    train_split, val_split = random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(tcfg["seed"])
    )
    # val split uses train mode (MLM masking applied in __getitem__)

    logger.info(f"Dataset: {n_train} train / {n_val} val sequences")

    train_dl = DataLoader(
        train_split,
        batch_size  = tcfg["batch_size"],
        shuffle     = True,
        num_workers = dcfg["num_workers"],
        pin_memory  = dcfg["pin_memory"] and torch.cuda.is_available(),
        drop_last   = True,
    )
    val_dl = DataLoader(
        val_split,
        batch_size  = tcfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = dcfg["num_workers"],
        pin_memory  = dcfg["pin_memory"] and torch.cuda.is_available(),
    )
    return train_dl, val_dl


# -----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.preset:
        cfg["model"]["preset"] = args.preset
        from src.config_loader import _resolve_model_preset
        cfg["model"] = _resolve_model_preset(cfg["model"])

    tcfg = cfg["training"]
    set_seed(tcfg["seed"])

    Path("outputs/logs").mkdir(parents=True, exist_ok=True)

    device = get_device(cfg)
    logger.info(f"Device: {device}")

    tokenizer = ProteinTokenizer(
        mask_prob        = cfg["tokenizer"]["mask_prob"],
        mask_token_prob  = cfg["tokenizer"]["mask_token_prob"],
        random_token_prob= cfg["tokenizer"]["random_token_prob"],
    )

    # update gradient_checkpointing from training config
    cfg["model"]["gradient_checkpointing"] = tcfg.get("gradient_checkpointing", False)

    model = build_model(cfg["model"], vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    # optional: torch.compile (PyTorch >= 2.0, good for server)
    if cfg["hardware"].get("compile_model") and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile…")
        model = torch.compile(model)

    # optional resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    train_dl, val_dl = build_dataloaders(cfg, tokenizer)

    # save config snapshot for reproducibility
    save_config(cfg, "outputs/logs/config_snapshot.yaml")

    trainer = Trainer(model, train_dl, val_dl, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
