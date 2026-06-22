"""
Scaling Experiment — Plan B
===========================
训练6个独立模型，每个阶段在上一阶段基础上新增一个物种。
固定每物种序列数 & 固定训练步数，唯一变量是训练数据的物种多样性。
最终画出「训练物种数 → 人类蛋白质MLM准确率」曲线，寻找涌现跳跃点。

Usage:
    python scaling_experiment.py               # 完整实验（下载+训练+评估）
    python scaling_experiment.py --skip-download  # 跳过下载（数据已存在）
    python scaling_experiment.py --plot-only   # 只画图（所有阶段已完成）
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from src.config_loader import load_config
from src.dataset import ProteinDataset
from src.emergence import compute_mlm_metrics
from src.model import build_model, ProteinLM
from src.tokenizer import ProteinTokenizer

# 数据下载复用
sys.path.insert(0, str(Path(__file__).parent))
from data.download_data import download_taxon, filter_fasta

Path("outputs/scaling").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/scaling/scaling.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# 数据准备
# -----------------------------------------------------------------------
def prepare_species_data(
    species_list: list[dict],
    seqs_per_taxon: int,
    raw_dir: Path,
    proc_dir: Path,
    min_len: int,
    max_len: int,
    seqs_overrides: dict[str, int] | None = None,
) -> list[Path]:
    """下载并过滤类群数据，返回 processed FASTA 路径列表。

    seqs_overrides: 按 taxon name 覆盖默认 seqs_per_taxon
                    （例如 {"arthropoda": 5000, "nematoda": 5000}）
    """
    seqs_overrides = seqs_overrides or {}
    paths = []
    for sp in species_list:
        proc_path = proc_dir / f"{sp['name']}_{sp['id']}.fasta"
        n_target  = seqs_overrides.get(sp["name"], seqs_per_taxon)
        exclude_id = sp.get("exclude_id")
        if not proc_path.exists():
            raw_path = download_taxon(
                taxon_id         = sp["id"],
                taxon_name       = sp["name"],
                out_dir          = raw_dir,
                max_seqs         = int(n_target * 1.5),  # 多下一些，过滤后保留目标数量
                min_len          = min_len,
                max_len          = max_len,
                exclude_taxon_id = exclude_id,
            )
            kept = filter_fasta(raw_path, proc_path, min_len, max_len, n_target)
            logger.info(f"  {sp['label']}: {kept}/{n_target} sequences saved → {proc_path}")
        else:
            logger.info(f"  {sp['label']}: already exists → {proc_path}")
        paths.append(proc_path)
    return paths


# -----------------------------------------------------------------------
# 固定 epoch 训练（带验证集早停）
# -----------------------------------------------------------------------
def train_fixed_epochs(
    model:         ProteinLM,
    dataset,
    max_epochs:    int,
    val_ratio:     float,
    patience:      int,
    lr:            float,
    batch_size:    int,
    device:        torch.device,
    autocast_dtype,
) -> dict:
    """训练固定 epoch 数，带验证集早停；返回训练统计字典。"""
    # 划分训练/验证集
    n_val   = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    # 调度器：warmup 占总步数 10%，最少 100 步
    total_steps  = len(train_dl) * max_epochs
    warmup_steps = max(100, total_steps // 10)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup    = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr * 0.01
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler    = torch.amp.GradScaler("cuda", enabled=False)

    best_val_loss      = float("inf")
    best_state         = None
    epochs_no_improve  = 0
    train_losses: list[float] = []

    for epoch in range(1, max_epochs + 1):
        # ---- 训练 ----
        model.train()
        ep_losses: list[float] = []
        for batch in train_dl:
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

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_losses.append(loss.item())

        # ---- 验证 ----
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_dl:
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
                val_losses.append(loss.item())

        train_loss = float(np.mean(ep_losses))
        val_loss   = float(np.mean(val_losses))
        train_losses.append(train_loss)

        logger.info(
            f"    Epoch {epoch}/{max_epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"    Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    # 恢复最优权重
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {
        "train_losses":      train_losses,
        "final_train_loss":  train_losses[-1] if train_losses else float("inf"),
        "best_val_loss":     best_val_loss,
        "epochs_trained":    len(train_losses),
    }


# -----------------------------------------------------------------------
# 单阶段实验
# -----------------------------------------------------------------------
def run_stage(
    stage_cfg:        dict,
    stage_idx:        int,
    exp_cfg:          dict,
    model_cfg:        dict,
    tokenizer_cfg:    dict,
    training_cfg:     dict,
    data_cfg:         dict,
    hardware_cfg:     dict,
    tokenizer:        ProteinTokenizer,
    human_dl:         DataLoader,
    random_model:     ProteinLM,
    device:           torch.device,
    autocast_dtype,
    out_dir:          Path,
) -> dict:
    stage_name = stage_cfg["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Stage {stage_idx+1}: {stage_name}")
    logger.info(f"  Species: {[s['label'] for s in stage_cfg['species']]}")
    logger.info(f"{'='*60}")

    # 检查是否已有结果
    result_path = out_dir / f"{stage_name}_result.json"
    if result_path.exists():
        logger.info(f"  Already completed, loading cached result.")
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    # 准备数据
    raw_dir  = Path(data_cfg["raw_dir"])
    proc_dir = Path(data_cfg["processed_dir"])
    fasta_paths = prepare_species_data(
        species_list   = stage_cfg["species"],
        seqs_per_taxon = exp_cfg["seqs_per_taxon"],
        raw_dir        = raw_dir,
        proc_dir       = proc_dir,
        min_len        = data_cfg["min_seq_len"],
        max_len        = data_cfg["max_seq_len"],
        seqs_overrides = exp_cfg.get("seqs_overrides", {}),
    )

    # 数据集
    dataset = ProteinDataset(
        fasta_paths = fasta_paths,
        tokenizer   = tokenizer,
        max_len     = data_cfg["max_seq_len"],
        mode        = "train",
    )
    logger.info(f"  Total sequences: {len(dataset)}")

    # 从随机初始化开始训练
    torch.manual_seed(42 + stage_idx)
    model = build_model(model_cfg, tokenizer.vocab_size).to(device)
    logger.info(f"  Model parameters: {model.num_parameters:,}")

    # 训练（固定 epoch，带验证集早停）
    train_info = train_fixed_epochs(
        model          = model,
        dataset        = dataset,
        max_epochs     = exp_cfg["max_epochs"],
        val_ratio      = exp_cfg["val_ratio"],
        patience       = exp_cfg["patience"],
        lr             = training_cfg["lr"],
        batch_size     = training_cfg["batch_size"],
        device         = device,
        autocast_dtype = autocast_dtype,
    )

    # 保存 checkpoint
    ckpt_path = out_dir / f"{stage_name}_model.pt"
    torch.save({"model": model.state_dict(), "stage": stage_cfg}, ckpt_path)

    # 评估
    logger.info(f"  Evaluating on human proteins…")
    human_metrics  = compute_mlm_metrics(model, human_dl, device, autocast_dtype)
    random_metrics = compute_mlm_metrics(random_model, human_dl, device, autocast_dtype)

    result = {
        "stage_name":    stage_name,
        "display":       stage_cfg["display"],
        "n_species":     len(stage_cfg["species"]),
        "species":       [s["label"] for s in stage_cfg["species"]],
        "total_seqs":    len(dataset),
        "epochs_trained":train_info["epochs_trained"],
        "final_loss":    train_info["final_train_loss"],
        "best_val_loss": train_info["best_val_loss"],
        "human_metrics": human_metrics,
        "random_metrics_at_this_stage": random_metrics,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"  Human MLM accuracy: {human_metrics['mlm_accuracy']:.4f} "
                f"(random: {random_metrics['mlm_accuracy']:.4f})")

    del model
    torch.cuda.empty_cache()
    return result


# -----------------------------------------------------------------------
# 可视化
# -----------------------------------------------------------------------
def plot_emergence_curve(results: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_species  = [r["n_species"]               for r in results]
    acc_trained= [r["human_metrics"]["mlm_accuracy"]              for r in results]
    acc_random = [r["random_metrics_at_this_stage"]["mlm_accuracy"] for r in results]
    ppl_trained= [r["human_metrics"]["perplexity"]                for r in results]
    labels     = [r["display"]                 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Scaling Experiment — Emergence Analysis", fontsize=13, fontweight="bold")

    # 按 stage 顺序对应的"是否动物界" —— 用于上色分界
    # 当前 6 stage：1=细菌 2=古菌 3=真菌 4=植物 5=无脊椎 6=啮齿
    # 前 4 = 非动物，后 2 = 动物
    n_nonanimal = 4
    animal_start_n = n_species[n_nonanimal] if n_nonanimal < len(n_species) else None

    # ---- 左图：准确率曲线 ----
    ax = axes[0]
    ax.plot(n_species, acc_trained, "o-", color="#4C72B0", linewidth=2,
            markersize=8, label="Trained model (human)")
    ax.plot(n_species, acc_random,  "s--", color="#DD8452", linewidth=1.5,
            markersize=6, label="Random init (baseline)")
    if animal_start_n is not None:
        boundary = (n_species[n_nonanimal-1] + animal_start_n) / 2
        ax.axvline(x=boundary, color="gray", linestyle=":", alpha=0.7)
        ax.text(boundary + 0.05, min(acc_trained) + (max(acc_trained)-min(acc_trained))*0.1,
                "<- Animals added", fontsize=9, color="gray")
    ax.set_xlabel("Number of training taxa", fontsize=11)
    ax.set_ylabel("Human protein MLM accuracy", fontsize=11)
    ax.set_title("MLM Accuracy vs. Training Taxa", fontsize=11)
    ax.set_xticks(n_species)
    ax.set_xticklabels([str(n) for n in n_species], fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 右图：困惑度曲线 ----
    ax = axes[1]
    ax.plot(n_species, ppl_trained, "o-", color="#55A868", linewidth=2, markersize=8)
    if animal_start_n is not None:
        ax.axvline(x=boundary, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of training taxa", fontsize=11)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=11)
    ax.set_title("Perplexity vs. Training Taxa", fontsize=11)
    ax.set_xticks(n_species)
    ax.set_xticklabels([str(n) for n in n_species], fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "emergence_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Emergence curve saved → {path}")

    # ---- 涌现得分柱状图 ----
    fig, ax = plt.subplots(figsize=(8, 4))
    improvements = [
        (t - r) / (r + 1e-9)
        for t, r in zip(acc_trained, acc_random)
    ]
    colors = ["#4C72B0" if i < n_nonanimal else "#DD8452" for i in range(len(improvements))]
    ax.bar(n_species, improvements, color=colors, edgecolor="white", width=0.6)
    if animal_start_n is not None:
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.6)
        ax.text(boundary + 0.05, max(improvements) * 0.95,
                "Non-animal -> Animal", fontsize=9, color="gray")
    ax.set_xlabel("Number of training taxa", fontsize=11)
    ax.set_ylabel("Relative improvement over random baseline", fontsize=11)
    ax.set_title("Emergence Score per Stage", fontsize=11)
    ax.set_xticks(n_species)
    ax.set_xticklabels([str(n) for n in n_species], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    patches = [
        mpatches.Patch(color="#4C72B0", label="Non-animal taxa"),
        mpatches.Patch(color="#DD8452", label="Animal taxa"),
    ]
    ax.legend(handles=patches)
    plt.tight_layout()
    path2 = out_dir / "emergence_score.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    logger.info(f"Emergence score saved → {path2}")


# -----------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="config/config.yaml")
    parser.add_argument("--skip-download",  action="store_true")
    parser.add_argument("--plot-only",      action="store_true")
    parser.add_argument("--max-stages",     type=int, default=None, help="只跑前N个阶段（用于快速验证）")
    parser.add_argument("--dry-run-epochs", type=int, default=None, help="覆盖训练epoch数（用于快速验证）")
    parser.add_argument("--download-only",  action="store_true", help="只下载所有阶段+人类数据后退出（用于离线服务器准备）")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    exp_cfg     = cfg["scaling_experiment"]
    model_cfg   = cfg["model"]
    tok_cfg     = cfg["tokenizer"]
    train_cfg   = cfg["training"]
    data_cfg    = cfg["data"]
    hw_cfg      = cfg["hardware"]

    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs/scaling").mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prec   = train_cfg["precision"]
    autocast_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]
    logger.info(f"Device: {device}  Precision: {prec}")

    # 分词器
    tokenizer = ProteinTokenizer(
        mask_prob         = tok_cfg["mask_prob"],
        mask_token_prob   = tok_cfg["mask_token_prob"],
        random_token_prob = tok_cfg["random_token_prob"],
    )

    # ---- 只画图模式 ----
    if args.plot_only:
        results = []
        for stage in exp_cfg["stages"]:
            rp = out_dir / f"{stage['name']}_result.json"
            if rp.exists():
                with open(rp, encoding="utf-8") as f:
                    results.append(json.load(f))
        if not results:
            logger.error("No stage results found. Run the experiment first.")
            sys.exit(1)
        plot_emergence_curve(results, out_dir)
        return

    # ---- 准备人类评估数据 ----
    human_sp   = exp_cfg["human_species"]
    raw_dir    = Path(data_cfg["raw_dir"])
    proc_dir   = Path(data_cfg["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        human_paths = prepare_species_data(
            species_list   = [human_sp],
            seqs_per_taxon = exp_cfg["human_eval_seqs"],
            raw_dir        = raw_dir,
            proc_dir       = proc_dir,
            min_len        = data_cfg["min_seq_len"],
            max_len        = data_cfg["max_seq_len"],
        )
    else:
        human_paths = [proc_dir / f"{human_sp['name']}_{human_sp['id']}.fasta"]

    # ---- 仅下载模式：把所有阶段需要的数据全下完后退出 ----
    if args.download_only:
        logger.info("=== Download-only mode: collecting all unique taxa across stages ===")
        seen, all_taxa = set(), []
        for stage in exp_cfg["stages"]:
            for sp in stage["species"]:
                if sp["id"] not in seen:
                    seen.add(sp["id"])
                    all_taxa.append(sp)
        logger.info(f"  {len(all_taxa)} unique taxa to download: "
                    f"{[s['label'] for s in all_taxa]}")
        prepare_species_data(
            species_list   = all_taxa,
            seqs_per_taxon = exp_cfg["seqs_per_taxon"],
            raw_dir        = raw_dir,
            proc_dir       = proc_dir,
            min_len        = data_cfg["min_seq_len"],
            max_len        = data_cfg["max_seq_len"],
            seqs_overrides = exp_cfg.get("seqs_overrides", {}),
        )
        logger.info("All data downloaded. Exiting (use --skip-download to train).")
        return

    human_ds = ProteinDataset(
        fasta_paths = human_paths,
        tokenizer   = tokenizer,
        max_len     = data_cfg["max_seq_len"],
        mode        = "train",
    )
    human_dl = DataLoader(
        human_ds,
        batch_size  = train_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = torch.cuda.is_available(),
    )
    logger.info(f"Human evaluation set: {len(human_ds)} sequences")

    # 随机基线模型（固定种子，全程不变）
    torch.manual_seed(0)
    random_model = build_model(model_cfg, tokenizer.vocab_size).to(device)
    logger.info(f"Random baseline model: {random_model.num_parameters:,} parameters")

    # 覆盖训练 epoch 数（dry-run 模式）
    if args.dry_run_epochs:
        exp_cfg["max_epochs"] = args.dry_run_epochs
        logger.info(f"[DRY RUN] max_epochs overridden to {args.dry_run_epochs}")

    stages = exp_cfg["stages"]
    if args.max_stages:
        stages = stages[:args.max_stages]
        logger.info(f"[DRY RUN] Running only first {args.max_stages} stage(s)")

    # ---- 逐阶段训练 ----
    all_results = []
    for i, stage in enumerate(stages):
        if not args.skip_download:
            pass  # prepare_species_data 会在 run_stage 里调用
        result = run_stage(
            stage_cfg     = stage,
            stage_idx     = i,
            exp_cfg       = exp_cfg,
            model_cfg     = model_cfg,
            tokenizer_cfg = tok_cfg,
            training_cfg  = train_cfg,
            data_cfg      = data_cfg,
            hardware_cfg  = hw_cfg,
            tokenizer     = tokenizer,
            human_dl      = human_dl,
            random_model  = random_model,
            device        = device,
            autocast_dtype= autocast_dtype,
            out_dir       = out_dir,
        )
        all_results.append(result)

    # ---- 汇总结果 ----
    summary_path = out_dir / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nAll results saved → {summary_path}")

    # ---- 画图 ----
    plot_emergence_curve(all_results, out_dir)

    # ---- 打印摘要 ----
    logger.info("\n" + "=" * 60)
    logger.info("  SCALING EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'物种数':>6}  {'物种':20}  {'人类准确率':>10}  {'随机基线':>10}  {'提升':>8}")
    logger.info(f"  {'-'*60}")
    for r in all_results:
        new_sp = r["species"][-1] if r["species"] else ""
        logger.info(
            f"  {r['n_species']:>6}  {new_sp:20}  "
            f"{r['human_metrics']['mlm_accuracy']:>10.4f}  "
            f"{r['random_metrics_at_this_stage']['mlm_accuracy']:>10.4f}  "
            f"{(r['human_metrics']['mlm_accuracy'] - r['random_metrics_at_this_stage']['mlm_accuracy']):>+8.4f}"
        )
    logger.info("=" * 60)


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()
