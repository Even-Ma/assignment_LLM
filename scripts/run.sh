#!/bin/bash

# 设置随机种子以保证实验可复现性
SEED=42

# 运行主训练脚本
# 你可以在这里调整参数
python src/main.py \
    --data_folder "data/gigaword_subset" \
    --num_epochs 30 \
    --batch_size 256 \
    --lr 1e-4 \
    --d_model 256 \
    --n_layers 3 \
    --n_heads 4 \
    --d_ff 1024 \
    --seq_len 120 \
    --seed $SEED

echo "Training finished. Check the 'results' directory for logs, model, and plots."