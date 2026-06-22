"""Compare small vs medium model emergence curves side by side.

Usage:
    python compare_models.py
    # outputs: outputs/scaling/comparison_small_vs_medium.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

SMALL_DIR  = Path("outputs/scaling_small_v1")
MEDIUM_DIR = Path("outputs/scaling")
OUT_PATH   = Path("outputs/scaling/comparison_small_vs_medium.png")


def load_results(folder: Path) -> list[dict]:
    with open(folder / "all_results.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    small  = load_results(SMALL_DIR)
    medium = load_results(MEDIUM_DIR)

    n_small  = [r["n_species"] for r in small]
    n_medium = [r["n_species"] for r in medium]
    acc_small  = [r["human_metrics"]["mlm_accuracy"] for r in small]
    acc_medium = [r["human_metrics"]["mlm_accuracy"] for r in medium]
    rand_med   = [r["random_metrics_at_this_stage"]["mlm_accuracy"] for r in medium]

    # per-stage delta (vs previous stage)
    delta_small  = [0] + [acc_small[i]  - acc_small[i-1]  for i in range(1, len(acc_small))]
    delta_medium = [0] + [acc_medium[i] - acc_medium[i-1] for i in range(1, len(acc_medium))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Small (16M) vs Medium (38M) — Emergence Comparison",
                 fontsize=13, fontweight="bold")

    # ---- 左图：accuracy curves ----
    ax = axes[0]
    ax.plot(n_small, acc_small, "o--", color="#888888", linewidth=2,
            markersize=7, label="Small (16M)")
    ax.plot(n_medium, acc_medium, "o-", color="#4C72B0", linewidth=2.5,
            markersize=8, label="Medium (38M)")
    ax.plot(n_medium, rand_med, "s:", color="#DD8452", linewidth=1.2,
            markersize=5, alpha=0.7, label="Random baseline")

    boundary = 5.0   # between stage4 (n=4) and stage5 (n=6)
    ax.axvline(x=boundary, color="gray", linestyle=":", alpha=0.5)
    ax.text(boundary + 0.05, 0.06, "<- Animals added", fontsize=9, color="gray")

    # 注释最高点
    ax.annotate(f"{max(acc_medium):.4f}", xy=(n_medium[-1], acc_medium[-1]),
                xytext=(n_medium[-1] - 0.5, acc_medium[-1] + 0.005),
                fontsize=9, color="#4C72B0")
    ax.annotate(f"{max(acc_small):.4f}", xy=(n_small[-1], acc_small[-1]),
                xytext=(n_small[-1] - 0.5, acc_small[-1] - 0.012),
                fontsize=9, color="#666666")

    ax.set_xlabel("Number of training taxa", fontsize=11)
    ax.set_ylabel("Human protein MLM accuracy", fontsize=11)
    ax.set_title("Accuracy Curve", fontsize=11)
    ax.set_xticks(n_medium)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ---- 右图：per-stage delta bars ----
    ax = axes[1]
    width = 0.4
    x_idx = list(range(len(n_medium)))

    bars_s = ax.bar([x - width/2 for x in x_idx], delta_small,
                    width=width, color="#AAAAAA", edgecolor="white",
                    label="Small (16M)")
    bars_m = ax.bar([x + width/2 for x in x_idx], delta_medium,
                    width=width, color="#4C72B0", edgecolor="white",
                    label="Medium (38M)")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"+{r['display'].lstrip('+ ').split()[0]}"
                        if r["stage_name"] != "stage1" else "base"
                        for r in medium],
                       rotation=20, fontsize=8, ha="right")
    ax.set_xlabel("Stage (new taxon added)", fontsize=11)
    ax.set_ylabel("Δ Accuracy vs previous stage", fontsize=11)
    ax.set_title("Per-stage Gain", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
