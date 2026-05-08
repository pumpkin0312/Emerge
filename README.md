# Emerge — Protein Language Model Emergence Study

利用低等生物（细菌、真菌）蛋白质序列训练语言模型，观察模型在人类蛋白质上的**涌现（Emergence）**现象。

---

## 研究思路

**涌现**是指随着训练数据多样性的增加，模型在某个阈值处性能突然跃升的现象（而非线性增长）。

本项目通过以下实验设计来捕捉这一信号：

- 训练6个**独立模型**，每个模型的训练数据在上一阶段基础上新增一个物种
- 固定每物种序列数（500条）与训练步数（3000步），唯一变量是**数据多样性**
- 在每个阶段结束后，零样本评估模型在**人类蛋白质**上的 MLM 准确率
- 若曲线在某阶段出现陡坡（尤其是加入真菌时），即为涌现候选信号

```
阶段1: E. coli（细菌1）
阶段2: + B. subtilis（细菌2）
阶段3: + S. aureus（细菌3）
阶段4: + S. cerevisiae（真菌1）← 关键节点，酵母与人类同源性高
阶段5: + A. niger（真菌2）
阶段6: + C. albicans（真菌3）
```

---

## 模型架构

- **类型**：BERT-style Transformer Encoder
- **训练目标**：Masked Language Modeling（MLM），掩码 15% 氨基酸残基
- **分词粒度**：单氨基酸 token，词表大小 25（20种标准氨基酸 + 5个特殊 token）
- **位置编码**：RoPE（Rotary Position Embedding）
- **可配置规格**：

| Preset | 参数量 | hidden_dim | layers | heads |
|--------|--------|-----------|--------|-------|
| tiny   | ~3M    | 128       | 4      | 4     |
| small  | ~16M   | 256       | 6      | 8     |
| medium | ~85M   | 512       | 12     | 16    |
| large  | ~340M  | 1024      | 24     | 16    |

---

## 项目结构

```
Emerge/
├── config/
│   └── config.yaml          # 所有超参数，在此修改
├── src/
│   ├── config_loader.py     # 配置加载
│   ├── tokenizer.py         # 氨基酸分词器 + MLM 掩码
│   ├── model.py             # Transformer 模型定义
│   ├── dataset.py           # PyTorch Dataset
│   ├── trainer.py           # 训练循环
│   └── emergence.py         # 涌现分析指标与可视化
├── data/
│   └── download_data.py     # 从 UniProt 下载蛋白质序列
├── scripts/
│   ├── quick_test.py        # 环境验证（无需数据）
│   └── profile_memory.py    # GPU 显存分析
├── train.py                 # 单模型训练入口
├── evaluate.py              # 单模型涌现评估
├── scaling_experiment.py    # Scaling Experiment 主脚本
└── environment.yaml         # Conda 环境配置
```

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yaml
conda activate emerge
```

> **服务器注意**：`environment.yaml` 中 PyTorch 默认使用 CUDA 12.1。
> 若服务器 CUDA 版本不同，修改以下行后再创建环境：
> ```yaml
> - torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
> ```
> 将 `cu121` 替换为实际版本（如 `cu118`、`cu124`）。

### 2. 验证环境

```bash
python scripts/quick_test.py
```

输出 `ALL TESTS PASSED` 且 `Device: cuda` 即为正常。

---

## 使用方法

### 快速验证（无需完整数据）

```bash
python scaling_experiment.py --max-stages 1 --dry-run-steps 100
```

### 完整 Scaling Experiment（正式实验）

```bash
python scaling_experiment.py
```

脚本会自动完成：
1. 从 UniProt 下载各物种数据（约 30-60 分钟）
2. 逐阶段独立训练（每阶段约 20-40 分钟，共 6 阶段）
3. 每阶段评估人类蛋白质 MLM 准确率
4. 输出涌现曲线图至 `outputs/scaling/`

**断点续传**：已完成的阶段会自动跳过，中断后可直接重新运行。

#### 其他参数

```bash
# 跳过数据下载（数据已存在时）
python scaling_experiment.py --skip-download

# 只重新画图（所有阶段已完成时）
python scaling_experiment.py --plot-only

# 只跑前N个阶段
python scaling_experiment.py --max-stages 3
```

### 单模型训练与评估

```bash
# 训练
python train.py

# 评估（涌现分析）
python evaluate.py
```

---

## 配置说明

所有超参数统一在 `config/config.yaml` 中修改，无需改动代码。

关键配置项：

```yaml
# 模型规格
model:
  preset: "small"   # tiny / small / medium / large

# Scaling Experiment
scaling_experiment:
  seqs_per_taxon: 10000      # 每个新增类群序列数
  human_eval_seqs: 5000      # 人类评估集大小
  max_epochs: 50             # 每阶段最多训练轮数
  patience: 5                # 早停 patience

# 训练
training:
  batch_size: 32
  precision: "bf16"          # fp32 / fp16 / bf16
  gradient_checkpointing: false  # medium/large 建议开启
```

---

## 输出文件

```
outputs/
├── checkpoints/             # train.py 的模型检查点
├── logs/                    # 训练日志
├── figures/                 # evaluate.py 的分析图
│   ├── representations.png  # UMAP 表示空间
│   ├── emergence_summary.png
│   └── emergence_results.json
└── scaling/                 # scaling_experiment.py 的结果
    ├── stage1_result.json   # 各阶段评估结果
    ├── all_results.json     # 汇总
    ├── emergence_curve.png  # 涌现曲线
    └── emergence_score.png  # 各阶段涌现得分
```

---

## 硬件需求

| 配置 | 推荐 Preset | 预计实验时间 |
|------|------------|-------------|
| RTX 4060 (8GB) | small | ~4-6 小时 |
| RTX 3090 (24GB) | medium | ~3-4 小时 |
| A100 (40GB) | medium/large | ~1-2 小时 |

---

## 依赖

- Python 3.11
- PyTorch >= 2.2 (CUDA)
- numpy, scipy, scikit-learn, matplotlib
- umap-learn（表示空间可视化）

详见 `environment.yaml`。
