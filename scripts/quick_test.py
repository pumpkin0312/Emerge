"""Quick sanity check — runs without any downloaded data.

Usage:
    python scripts/quick_test.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config_loader import load_config
from src.tokenizer import ProteinTokenizer
from src.model import build_model

def main():
    print("=" * 55)
    print("  Emerge Quick Test")
    print("=" * 55)

    # 1. Config
    cfg = load_config()
    print(f"[1] Config loaded: preset={cfg['model']['preset']}")

    # 2. Tokenizer
    tok = ProteinTokenizer()
    sample = "MKTIIALSYIFCLVFA"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    assert decoded == sample, f"Tokenizer mismatch: {decoded} != {sample}"
    print(f"[2] Tokenizer OK: vocab_size={tok.vocab_size}")

    # 3. MLM masking
    masked, labels = tok.apply_mlm_mask(ids)
    n_masked = sum(1 for l in labels if l != -100)
    print(f"[3] MLM masking OK: {n_masked}/{len(ids)} tokens masked")

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = build_model(cfg["model"], tok.vocab_size).to(device)
    print(f"[5] Model: {model.num_parameters:,} parameters")

    # 5. Forward pass
    batch = tok.batch_encode_with_mlm([sample, "ACDEFGHIKLMNPQRSTVWY"], max_len=32)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    t0  = time.time()
    out = model(input_ids, attention_mask, output_attentions=True)
    dt  = (time.time() - t0) * 1000

    assert out["logits"].shape == (*input_ids.shape, tok.vocab_size)
    assert "attentions" in out
    print(f"[6] Forward pass OK: logits={tuple(out['logits'].shape)}  ({dt:.1f} ms)")

    # 6. Loss
    import torch.nn.functional as F
    loss = F.cross_entropy(
        out["logits"].view(-1, tok.vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    print(f"[7] MLM loss: {loss.item():.4f}")

    # 7. Representations
    repr_vec = model.get_representations(input_ids, attention_mask)
    assert repr_vec.shape == (2, cfg["model"]["hidden_dim"])
    print(f"[8] Representation shape: {tuple(repr_vec.shape)}  ✓")

    print("=" * 55)
    print("  ALL TESTS PASSED")
    print("  Next step: python data/download_data.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
