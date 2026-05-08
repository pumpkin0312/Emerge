"""Amino-acid level tokenizer for protein sequences."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch


# 20 standard amino acids
_AA = list("ACDEFGHIKLMNPQRSTVWY")

_SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[EOS]", "[MASK]"]

PAD_ID  = 0
UNK_ID  = 1
CLS_ID  = 2
EOS_ID  = 3
MASK_ID = 4


@dataclass
class ProteinTokenizer:
    mask_prob:         float = 0.15
    mask_token_prob:   float = 0.80
    random_token_prob: float = 0.10
    # keep_prob = 1 - mask_token_prob - random_token_prob

    vocab:    list[str]      = field(init=False)
    token2id: dict[str, int] = field(init=False)
    id2token: dict[int, str] = field(init=False)

    def __post_init__(self):
        self.vocab    = _SPECIAL + _AA
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def aa_token_ids(self) -> list[int]:
        return [self.token2id[aa] for aa in _AA]

    # ------------------------------------------------------------------
    def encode(self, seq: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token2id.get(aa, UNK_ID) for aa in seq.upper()]
        if add_special_tokens:
            ids = [CLS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = [self.id2token.get(i, "?") for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in _SPECIAL]
        return "".join(tokens)

    # ------------------------------------------------------------------
    def apply_mlm_mask(
        self,
        input_ids: list[int],
        rng: Optional[random.Random] = None,
    ) -> tuple[list[int], list[int]]:
        """Return (masked_ids, labels).

        labels[i] == -100 means position is not masked (ignored in loss).
        """
        rng = rng or random
        masked   = input_ids[:]
        labels   = [-100] * len(input_ids)

        aa_ids = set(self.aa_token_ids)
        for i, tok in enumerate(input_ids):
            if tok not in aa_ids:
                continue
            if rng.random() >= self.mask_prob:
                continue

            labels[i] = tok
            r = rng.random()
            if r < self.mask_token_prob:
                masked[i] = MASK_ID
            elif r < self.mask_token_prob + self.random_token_prob:
                masked[i] = rng.choice(self.aa_token_ids)
            # else: keep original

        return masked, labels

    # ------------------------------------------------------------------
    def batch_encode_with_mlm(
        self,
        sequences: list[str],
        max_len: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenise, pad, and apply MLM masking for a batch of sequences."""
        all_input_ids, all_labels, all_attention = [], [], []
        rng = random.Random()

        for seq in sequences:
            ids = self.encode(seq, add_special_tokens=True)
            ids = ids[:max_len]

            masked, labels = self.apply_mlm_mask(ids, rng)

            pad_len = max_len - len(masked)
            attn    = [1] * len(masked) + [0] * pad_len
            masked  = masked  + [PAD_ID]  * pad_len
            labels  = labels  + [-100]    * pad_len

            all_input_ids.append(masked)
            all_labels.append(labels)
            all_attention.append(attn)

        return {
            "input_ids":      torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention, dtype=torch.long),
            "labels":         torch.tensor(all_labels,    dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def batch_encode(
        self,
        sequences: list[str],
        max_len: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenise and pad without masking (for evaluation)."""
        all_input_ids, all_attention = [], []

        for seq in sequences:
            ids = self.encode(seq, add_special_tokens=True)
            ids = ids[:max_len]
            pad_len = max_len - len(ids)
            attn    = [1] * len(ids) + [0] * pad_len
            ids     = ids + [PAD_ID] * pad_len

            all_input_ids.append(ids)
            all_attention.append(attn)

        return {
            "input_ids":      torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention, dtype=torch.long),
        }
