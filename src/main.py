import argparse
from train import train_model

def get_config():
    """
    获取训练配置。
    """
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    
    # 数据相关参数
    parser.add_argument('--data_folder', type=str, default='data/gigaword_subset', help='Folder with data files')
    parser.add_argument('--train_src_file', type=str, default='data/gigaword_subset/train.src.txt', help='Training source file')
    parser.add_argument('--train_tgt_file', type=str, default='data/gigaword_subset/train.tgt.txt', help='Training target file')
    parser.add_argument('--val_src_file', type=str, default='data/gigaword_subset/dev.src.txt', help='Validation source file')
    parser.add_argument('--val_tgt_file', type=str, default='data/gigaword_subset/dev.tgt.txt', help='Validation target file')

    # 模型超参数
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=100, help='Max sequence length')

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for AdamW')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    config = vars(args)
    return config

if __name__ == '__main__':
    config = get_config()
    import torch
    torch.manual_seed(config['seed'])
    train_model(config)