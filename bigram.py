"""
Bigram 字符级语言模型
=====================
最简单的语言模型：用一个字符预测下一个字符。
本质是一张 vocab_size × vocab_size 的查找表（lookup table），
每个位置存的是"看到字符 i 时，下一个字符是 j 的 logit（未归一化得分）"。

参数量极小（约 729 个），但完全没有上下文能力。
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 从 makemore.py 导入通用工具（纯函数/类，无全局变量依赖）
from makemore import ModelConfig, CharDataset, create_datasets, InfiniteDataLoader, generate

# =============================================================================
# Bigram 语言模型
# =============================================================================

class Bigram(nn.Module):
    """
    Bigram 语言模型：本质上就是一张可学习的查找表。

    结构：
        self.logits: shape (vocab_size, vocab_size) 的可学习参数矩阵
        - 行索引：当前字符（输入）
        - 列索引：下一个字符（预测目标）
        - 值：该转移的 logit（经 softmax 后变成概率）

    前向传播：
        给定输入 idx（字符 index），直接用它在矩阵里查行，
        得到对应的下一个字符的 logit 分布。
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        # 核心参数：一张 n×n 的表，初始化为全零
        # 训练时梯度会更新这张表，让常见的字符转移得分更高
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        # Bigram 只需要看前 1 个字符，所以 block_size = 1
        return 1

    def forward(self, idx, targets=None):
        """
        参数：
            idx     : (B, T) 整数张量，每个元素是字符的 index
            targets : (B, T) 整数张量，每个位置对应的"正确下一个字符"

        返回：
            logits : (B, T, vocab_size) 每个位置上，下一个字符的 logit
            loss   : 标量，cross-entropy 损失（若 targets 为 None 则返回 None）
        """
        # 直接用 idx 当行索引，从表里取出对应行
        # idx shape: (B, T)  →  logits shape: (B, T, vocab_size)
        logits = self.logits[idx]

        loss = None
        if targets is not None:
            # cross_entropy 要求输入 shape 是 (N, C)，所以先 view 成 (-1, vocab_size)
            # ignore_index=-1 表示 targets 中值为 -1 的位置不参与 loss 计算（padding 位置）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

# =============================================================================
# 依赖全局变量的辅助函数（无法从 makemore.py 直接 import）
# =============================================================================

def print_samples(num=10):
    """从模型中采样并打印，统计哪些是新名字、哪些已在训练/测试集里"""
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()  # 去掉开头的 <START> token
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)

    print('-' * 80)
    for lst, desc in [(train_samples, '训练集已有'), (test_samples, '测试集已有'), (new_samples, '全新生成')]:
        print(f"{len(lst)} 个样本属于【{desc}】:")
        for word in lst:
            print(word)
    print('-' * 80)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    """在给定数据集上评估平均 loss"""
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]  # args.device 来自全局
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Bigram 语言模型")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt')
    parser.add_argument('--work-dir',   '-o', type=str, default='out_bigram')
    parser.add_argument('--resume',     action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--num-workers', '-n', type=int, default=4)
    parser.add_argument('--max-steps',  type=int, default=-1)
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--seed',       type=int, default=3407)
    parser.add_argument('--top-k',      type=int, default=-1)
    parser.add_argument('--batch-size', '-b', type=int,   default=32)
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4)
    parser.add_argument('--weight-decay',  '-w', type=float, default=0.01)
    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # 准备数据
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"vocab_size={vocab_size}, block_size={block_size}")

    # 初始化模型（Bigram 只用 vocab_size，其余超参数忽略）
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size)
    model = Bigram(config)
    model.to(args.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    # Bigram 参数量 = vocab_size × vocab_size = 27×27 = 729（对于名字数据集）

    if args.resume or args.sample_only:
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8
    )
    batch_loader = InfiniteDataLoader(
        train_dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers
    )

    # 训练循环
    best_loss = None
    step = 0
    while True:
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # ── 前向传播 ──────────────────────────────────────────────────────────
        # model(X, Y) 内部做两件事：
        #   1. logits = self.logits[X]：用 X 里每个字符的 index 查表，
        #      得到该字符对应的"下一个字符得分"，shape (B, T, vocab_size)
        #   2. loss = cross_entropy(logits, Y)：
        #      对每个位置，loss = -log( softmax(logits)[正确字符] )
        #      预测正确字符的概率越高，loss 越小
        logits, loss = model(X, Y)

        # ── 反向传播 ──────────────────────────────────────────────────────────
        # zero_grad：清除上一步残留的梯度（PyTorch 默认累加梯度）
        model.zero_grad(set_to_none=True)
        # loss.backward()：自动计算 loss 对 self.logits 矩阵每个元素的梯度
        #   - logits[i][j] 偏低（预测 i→j 概率不够高）→ 梯度为正 → 下一步调大
        #   - logits[i][j] 偏高（预测 i→j 概率太高）  → 梯度为负 → 下一步调小
        loss.backward()
        # optimizer.step()：用梯度更新参数
        #   logits[i][j] -= lr * gradient
        # 训练完成后，矩阵收敛到训练集中字符转移频率的近似：
        #   例如 softmax(logits[a]) 中 n/r/l 概率高（an/ar/al 常见），z 概率极低
        optimizer.step()

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test",  test_loss,  step)
            writer.flush()
            print(f"step {step} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f}")
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"保存模型到 {out_path}（test loss: {test_loss:.4f}）")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        if args.max_steps >= 0 and step >= args.max_steps:
            break
