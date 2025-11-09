import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import List, Tuple
from pathlib import Path

class BilingualDataset(Dataset):
    """
    自定义数据集类，用于处理双语平行语料。
    """
    def __init__(self, ds_path, src_lang_tokenizer, tgt_lang_tokenizer, src_vocab, tgt_vocab, src_seq_len, tgt_seq_len):
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.ds_path = ds_path
        self.src_lang_tokenizer = src_lang_tokenizer
        self.tgt_lang_tokenizer = tgt_lang_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.sos_token = torch.tensor([src_vocab["<SOS>"]], dtype=torch.int64)
        self.eos_token = torch.tensor([src_vocab["<EOS>"]], dtype=torch.int64)
        self.pad_token = torch.tensor([src_vocab["<PAD>"]], dtype=torch.int64)

        with open(ds_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        src_text, tgt_text = line.strip().split('\t')

        src_tokens = self.src_lang_tokenizer(src_text)
        tgt_tokens = self.tgt_lang_tokenizer(tgt_text)

        src_indices = self.src_vocab.lookup_indices(src_tokens)
        tgt_indices = self.tgt_vocab.lookup_indices(tgt_tokens)

        # 编码器输入
        enc_input_tokens = src_indices[:self.src_seq_len - 2]
        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * (self.src_seq_len - len(enc_input_tokens) - 2), dtype=torch.int64),
            ],
            dim=0,
        )

        # 解码器输入
        dec_input_tokens = tgt_indices[:self.tgt_seq_len - 1]
        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * (self.tgt_seq_len - len(dec_input_tokens) - 1), dtype=torch.int64),
            ],
            dim=0,
        )
        
        # 标签（模型期望的输出）
        label_tokens = tgt_indices[:self.tgt_seq_len - 1]
        label = torch.cat(
            [
                torch.tensor(label_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * (self.tgt_seq_len - len(label_tokens) - 1), dtype=torch.int64),
            ],
            dim=0,
        )

        assert enc_input.size(0) == self.src_seq_len
        assert dec_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    """
    创建因果遮罩，防止解码器在预测时看到未来的token。
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_or_build_tokenizer(lang: str) -> get_tokenizer:
    """
    获取或构建分词器。这里使用最基础的空格分词。
    """
    return get_tokenizer('basic_english')

def yield_tokens(file_path: str, tokenizer, lang_idx: int):
    """
    从文件中逐行读取并生成 token。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')[lang_idx]
            yield tokenizer(text)

def get_all_sentences(ds_path: str, lang_idx: int):
    """
    获取数据集中所有的句子。
    """
    with open(ds_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().split('\t')[lang_idx]


def get_vocab(config):
    """
    构建源语言和目标语言的词表。
    """
    # 将训练、验证和测试集合并，以构建完整的词表
    # 这是一种简化处理，实际项目中可能会分开处理或使用预训练词表
    combined_train_path = Path(f"{config['data_folder']}/combined_train.tsv")
    combined_val_path = Path(f"{config['data_folder']}/combined_val.tsv")
    
    # 合并文件
    if not combined_train_path.exists():
        with open(combined_train_path, 'w', encoding='utf-8') as outfile:
            with open(config['train_src_file'], 'r', encoding='utf-8') as f_src, open(config['train_tgt_file'], 'r', encoding='utf-8') as f_tgt:
                for src_line, tgt_line in zip(f_src, f_tgt):
                    outfile.write(f"{src_line.strip()}\t{tgt_line.strip()}\n")

    if not combined_val_path.exists():
        with open(combined_val_path, 'w', encoding='utf-8') as outfile:
            with open(config['val_src_file'], 'r', encoding='utf-8') as f_src, open(config['val_tgt_file'], 'r', encoding='utf-8') as f_tgt:
                for src_line, tgt_line in zip(f_src, f_tgt):
                    outfile.write(f"{src_line.strip()}\t{tgt_line.strip()}\n")
    
    src_tokenizer = get_or_build_tokenizer('en')
    tgt_tokenizer = get_or_build_tokenizer('en')
    
    special_symbols = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    vocab_src = build_vocab_from_iterator(
        yield_tokens(str(combined_train_path), src_tokenizer, 0),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_src.set_default_index(vocab_src["<UNK>"])

    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(str(combined_train_path), tgt_tokenizer, 1),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_tgt.set_default_index(vocab_tgt["<UNK>"])
    
    return vocab_src, vocab_tgt, src_tokenizer, tgt_tokenizer

def get_ds(config):
    """
    获取数据集加载器。
    """
    vocab_src, vocab_tgt, src_tokenizer, tgt_tokenizer = get_vocab(config)

    train_ds_path = Path(f"{config['data_folder']}/combined_train.tsv")
    val_ds_path = Path(f"{config['data_folder']}/combined_val.tsv")

    train_ds = BilingualDataset(
        str(train_ds_path), src_tokenizer, tgt_tokenizer, vocab_src, vocab_tgt, 
        config['seq_len'], config['seq_len']
    )
    val_ds = BilingualDataset(
        str(val_ds_path), src_tokenizer, tgt_tokenizer, vocab_src, vocab_tgt, 
        config['seq_len'], config['seq_len']
    )

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, vocab_src, vocab_tgt