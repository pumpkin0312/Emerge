"""Protein Language Model — BERT-style Transformer Encoder with MLM head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Positional Encoding
# -----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — better than learned abs position."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# -----------------------------------------------------------------------
# Multi-Head Self-Attention
# -----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope  = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d)

        q, k = self.rope(q, k)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        if attention_mask is not None:
            # mask: (B, L) → (B, 1, 1, L)
            mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            attn = attn + mask

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out(out), attn_weights


# -----------------------------------------------------------------------
# Feed-Forward Network
# -----------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------
# Transformer Encoder Layer
# -----------------------------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn   = FFN(hidden_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN (more stable training)
        attn_out, attn_weights = self.attn(self.norm1(x), attention_mask)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


# -----------------------------------------------------------------------
# MLM Head
# -----------------------------------------------------------------------
class MLMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.dense  = nn.Linear(hidden_dim, hidden_dim)
        self.act    = nn.GELU()
        self.norm   = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(x))))


# -----------------------------------------------------------------------
# Protein Language Model
# -----------------------------------------------------------------------
class ProteinLM(nn.Module):
    def __init__(
        self,
        vocab_size:   int,
        hidden_dim:   int   = 256,
        num_layers:   int   = 6,
        num_heads:    int   = 8,
        ffn_dim:      int   = 1024,
        dropout:      float = 0.1,
        max_position: int   = 512,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = gradient_checkpointing

        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlm_head = MLMHead(hidden_dim, vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.weight[0])  # PAD token

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self.embed_drop(self.embed(input_ids))

        all_attentions = []
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x, attn = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_reentrant=False
                )
            else:
                x, attn = layer(x, attention_mask)
            if output_attentions:
                all_attentions.append(attn)

        x = self.norm(x)
        logits = self.mlm_head(x)

        result = {"logits": logits, "last_hidden_state": x}
        if output_attentions:
            result["attentions"] = torch.stack(all_attentions, dim=1)  # (B, L, H, T, T)
        return result

    # ------------------------------------------------------------------
    def get_representations(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_index:    int = -1,
    ) -> torch.Tensor:
        """Return per-sequence mean-pool representation from a given layer."""
        x = self.embed_drop(self.embed(input_ids))
        layers = self.layers
        if layer_index < 0:
            layer_index = len(layers) + layer_index

        for i, layer in enumerate(layers):
            x, _ = layer(x, attention_mask)
            if i == layer_index:
                break
        x = self.norm(x)

        # Mean pool over non-padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return x.mean(1)

    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------
def build_model(model_cfg: dict, vocab_size: int) -> ProteinLM:
    return ProteinLM(
        vocab_size   = vocab_size,
        hidden_dim   = model_cfg["hidden_dim"],
        num_layers   = model_cfg["num_layers"],
        num_heads    = model_cfg["num_heads"],
        ffn_dim      = model_cfg["ffn_dim"],
        dropout      = model_cfg["dropout"],
        max_position = model_cfg["max_position"],
        gradient_checkpointing = model_cfg.get("gradient_checkpointing", False),
    )
