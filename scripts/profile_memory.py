"""Profile GPU memory usage at different model presets and batch sizes.

Helps choose the right config before committing to a long training run.

Usage:
    python scripts/profile_memory.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config_loader import load_config, _resolve_model_preset
from src.model import build_model
from src.tokenizer import ProteinTokenizer

PRESETS     = ["tiny", "small", "medium"]
BATCH_SIZES = [8, 16, 32, 64]
SEQ_LEN     = 512


def measure(preset: str, batch_size: int, device: torch.device):
    cfg = load_config()
    cfg["model"]["preset"] = preset
    cfg["model"] = _resolve_model_preset(cfg["model"])
    tok = ProteinTokenizer()

    model = build_model(cfg["model"], tok.vocab_size).to(device)

    seq = "A" * SEQ_LEN
    batch = tok.batch_encode_with_mlm([seq] * batch_size, max_len=SEQ_LEN)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out  = model(input_ids, attention_mask)
            import torch.nn.functional as F
            loss = F.cross_entropy(
                out["logits"].view(-1, tok.vocab_size),
                labels.view(-1), ignore_index=-100
            )
            loss.backward()
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        return mem, "OK"
    except torch.cuda.OutOfMemoryError:
        return None, "OOM"
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found. Run on a GPU machine.")
        return

    device = torch.device("cuda")
    total  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {total:.1f} GB")
    print(f"{'Preset':<10} {'Batch':>6} {'VRAM (GB)':>12} {'Status':>8}")
    print("-" * 42)

    for preset in PRESETS:
        for bs in BATCH_SIZES:
            mem, status = measure(preset, bs, device)
            mem_str = f"{mem:.2f}" if mem else "—"
            ok = "✓" if status == "OK" else "✗ OOM"
            print(f"{preset:<10} {bs:>6} {mem_str:>12} {ok:>8}")


if __name__ == "__main__":
    main()
