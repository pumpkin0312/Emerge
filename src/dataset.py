"""PyTorch Dataset for protein sequences."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ProteinTokenizer


# -----------------------------------------------------------------------
class ProteinDataset(Dataset):
    def __init__(
        self,
        fasta_paths: list[str | Path],
        tokenizer:   ProteinTokenizer,
        max_len:     int  = 512,
        mode:        str  = "train",  # "train" | "eval"
        shuffle_seqs: bool = False,   # for shuffled-sequence baseline
        random_window: bool = False,  # 随机子窗口采样（避免 N 端偏置，仅小窗口实验需要）
    ):
        self.tokenizer    = tokenizer
        self.max_len      = max_len
        self.mode         = mode
        self.shuffle_seqs = shuffle_seqs
        self.random_window= random_window

        self.sequences: list[str] = []
        for path in fasta_paths:
            self.sequences.extend(_parse_fasta(path))

        if not self.sequences:
            raise RuntimeError(f"No sequences found in {fasta_paths}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        if self.shuffle_seqs:
            seq = _shuffle_seq(seq)

        # 随机窗口采样：仅当序列长度超过窗口、且开启 random_window 时生效
        # 保留 2 个位置给 CLS / EOS
        inner_max = self.max_len - 2
        if self.random_window and len(seq) > inner_max:
            start = random.randint(0, len(seq) - inner_max)
            seq = seq[start : start + inner_max]

        ids = self.tokenizer.encode(seq, add_special_tokens=True)
        ids = ids[: self.max_len]

        if self.mode == "train":
            masked, labels = self.tokenizer.apply_mlm_mask(ids)
        else:
            masked, labels = ids[:], [-100] * len(ids)

        pad_len = self.max_len - len(masked)
        attn    = [1] * len(masked) + [0] * pad_len
        masked  = masked + [0] * pad_len
        labels  = labels + [-100] * pad_len

        return {
            "input_ids":      torch.tensor(masked, dtype=torch.long),
            "attention_mask": torch.tensor(attn,   dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


# -----------------------------------------------------------------------
def _parse_fasta(path: str | Path) -> list[str]:
    """Read sequences from a FASTA file."""
    sequences = []
    current   = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line.upper())
    if current:
        sequences.append("".join(current))
    return sequences


def _shuffle_seq(seq: str) -> str:
    aa = list(seq)
    random.shuffle(aa)
    return "".join(aa)


# -----------------------------------------------------------------------
def split_dataset(
    sequences: list[str],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]
