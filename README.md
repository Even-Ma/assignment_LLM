# 从零实现 Transformer (Transformer from Scratch)

## 项目简介

本项目旨在根据课程作业要求，从零开始手工搭建一个完整的 Transformer 模型。我们将在 Gigaword 数据集的子集上进行文本摘要任务的训练和验证。

该实现包含了 Transformer 的核心组件：
- **多头自注意力机制 (Multi-Head Self-Attention)**
- **位置前馈网络 (Position-wise Feed-Forward Networks)**
- **残差连接与层归一化 (Residual Connections & Layer Normalization)**
- **位置编码 (Positional Encoding)**
- **完整的 Encoder-Decoder 架构**

此外，项目还实现了多项训练稳定性技巧，包括：
- **AdamW 优化器**
- **学习率动态调度 (Warm-up and decay)**
- **梯度裁剪 (Gradient Clipping)**
- **模型保存与加载**
- **训练过程可视化**

## 项目结构

```
.
├── README.md             # 本文件
├── requirements.txt      # Python 依赖
├── scripts/
│   └── run.sh            # 训练脚本
├── src/
│   ├── model.py          # Transformer 模型定义
│   ├── dataset.py        # 数据加载与预处理
│   ├── train.py          # 训练与验证逻辑
│   ├── utils.py          # 辅助函数
│   └── main.py           # 主程序入口
└── results/              # (自动创建) 存放日志、模型和结果图
```

## 环境与硬件要求

### 1. 硬件要求

- **CPU**: 任意现代多核 CPU 均可。
- **内存**: 建议至少 16GB RAM。
- **GPU**: 强烈建议使用 NVIDIA GPU（CUDA 支持）以加速训练。至少需要 8GB 显存。如果无 GPU，训练会非常缓慢。

### 2. 环境设置

首先，克隆本仓库：
```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

然后，创建虚拟环境并安装依赖：
```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # on Windows, use `venv\Scripts\activate`

# 安装依赖
pip install -r requirements.txt

# 下载 Spacy 英语模型 (用于分词)
python -m spacy download en_core_web_sm
```

### 3. 数据准备

本项目使用 Gigaword 数据集。请按照以下结构组织数据文件：
```
.
└── data/
    └── org_data/
        ├── train.src.txt  # 训练集源文本
        ├── train.tgt.txt  # 训练集目标文本
        ├── dev.src.txt    # 验证集源文本
        ├── dev.tgt.txt    # 验证集目标文本
        └── ...
```
代码将从 `data/org_data` 目录加载数据。

## 如何运行

### 一键式训练

我们提供了一个 `run.sh` 脚本来复现实验。此脚本包含了所有必要的命令行参数和固定的随机种子。

**要开始训练，请执行：**
```bash
bash scripts/run.sh
```

### 复现实验的精确命令行

`run.sh` 脚本中执行的命令如下，你可以手动运行此命令以获得相同的结果：

```bash
python src/main.py \
    --num_epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 4 \
    --d_ff 1024 \
    --seq_len 120 \
    --seed 42
```

```markdown
# 从零实现 Transformer (支持 DDP 多卡训练)

本仓库实现了一个简易的 Transformer（Encoder-Decoder）并支持 PyTorch DDP（Distributed Data Parallel）在多卡上并行训练。

重要说明（DDP 相关）：
- 使用 torchrun 启动多进程训练（每个进程绑定到一个 GPU）。
- run.sh 提供了启动 4 卡训练的示例命令。
- config 中的 --batch_size 表示“每个 GPU（per-process）的批大小”。总批大小 = batch_size * nproc_per_node。
- 训练/验证日志与模型保存只在 rank 0 进程（主进程）进行，以避免重复写文件。

如何在 4 卡上运行（精确命令）：
```bash
# 假设你有 4 张可用 GPU，使用 torchrun 启动 4 个进程（每进程绑定到一张 GPU）
torchrun --nproc_per_node=4 python src/main.py \
    --num_epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 4 \
    --d_ff 1024 \
    --seq_len 120 \
    --seed 42
```
说明：这里的 --batch_size 16 表示每 GPU 的 batch 是 16，若用 4 卡，总 batch = 16 * 4 = 64。

run.sh（已提供）也会运行类似的 torchrun 命令以便一键启动。

其他使用细节与文件结构请参考仓库中原 README 的其余内容（未改动的部分）。
```

### 结果查看

训练过程中：
- **日志**: 训练日志（包括损失和学习率）会实时打印在控制台，并同时保存在 `results/training.log` 文件中。
- **模型**: 验证损失最低的模型权重将保存在 `results/best_model.pt`。
- **图表**: 训练结束后，训练和验证损失曲线图将保存在 `results/loss_curves.png`。

