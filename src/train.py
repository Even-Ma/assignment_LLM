import torch
import torch.nn as nn
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import logging

from model import build_transformer
from dataset import get_ds, causal_mask
from utils import count_parameters, get_lr_scheduler, plot_training_curves

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    构建模型。
    """
    model = build_transformer(
        vocab_src_len, vocab_tgt_len,
        config['seq_len'], config['seq_len'],
        d_model=config['d_model'],
        N=config['n_layers'],
        h=config['n_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )
    return model

def train_one_epoch(model, train_dataloader, optimizer, scheduler, criterion, device, epoch, log_file):
    """
    训练一个 epoch。
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Training]")
    for batch in pbar:
        optimizer.zero_grad()

        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)

        loss = criterion(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

    avg_loss = total_loss / len(train_dataloader)
    logging.info(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, val_dataloader, criterion, device, epoch, log_file):
    """
    验证一个 epoch。
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Validation]")
        for batch in pbar:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = criterion(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_dataloader)
    logging.info(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train_model(config):
    """
    完整的训练流程。
    """
    # 设置日志
    log_dir = 'results'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 获取数据
    train_dataloader, val_dataloader, vocab_src, vocab_tgt = get_ds(config)
    vocab_src_len = len(vocab_src)
    vocab_tgt_len = len(vocab_tgt)
    logging.info(f"Source vocab size: {vocab_src_len}, Target vocab size: {vocab_tgt_len}")

    # 构建模型
    model = get_model(config, vocab_src_len, vocab_tgt_len).to(device)
    logging.info(f"Model created with {count_parameters(model):,} trainable parameters.")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=4000, d_model=config['d_model'])
    
    # 在损失函数中忽略填充 token
    pad_idx = vocab_tgt['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, criterion, device, epoch, log_file)
        val_loss = validate_one_epoch(model, val_dataloader, criterion, device, epoch, log_file)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(log_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, model_save_path)
            logging.info(f"Best model saved to {model_save_path}")

    # 绘制损失曲线
    plot_training_curves(train_losses, val_losses, log_dir)
    logging.info("Training complete.")
