import torch
import matplotlib.pyplot as plt
import os

def count_parameters(model):
    """
    计算模型的可训练参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr_scheduler(optimizer, warmup_steps, d_model):
    """
    根据 "Attention Is All You Need" 论文中的公式创建学习率调度器。
    """
    def lr_lambda(step):
        step += 1 # step 从 1 开始
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_training_curves(train_losses, val_losses, log_dir):
    """
    绘制并保存训练和验证损失曲线。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(log_dir, 'loss_curves.png')
    plt.savefig(save_path)
    print(f"Loss curves saved to {save_path}")
    plt.close()