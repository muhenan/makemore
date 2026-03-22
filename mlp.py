"""
MLP 字符级语言模型
==================
基于 Bengio et al. 2003 的经典论文：
    "A Neural Probabilistic Language Model"
    https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

核心思路：
    - 不像 Bigram 只看前 1 个字符，MLP 看前 block_size 个字符
    - 每个字符先通过 Embedding 表映射成一个低维向量
    - 把 block_size 个向量拼接在一起，送入一个两层 MLP
    - MLP 输出下一个字符的 logit 分布

对比 Bigram 的进步：
    - Bigram：只有 1 个字符的上下文，参数量 ~729
    - MLP：有 block_size 个字符的上下文，能学到字符组合规律，参数量更多但能力更强
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
# MLP 语言模型
# =============================================================================

class MLP(nn.Module):
    """
    MLP 语言模型详解：

    输入：idx，shape (B, T)
    输出：logits，shape (B, T, vocab_size)

    内部数据流（以 block_size=3, n_embd=64 为例）：

    1. 查 Embedding 表
       idx (B, T) → tok_emb (B, T, 64)

    2. 滑动窗口拼接（看前 block_size 个字符）
       对每个位置，把前 block_size 个 token 的 embedding 拼在一起
       得到 (B, T, 64*3) = (B, T, 192)

    3. MLP 前向
       Linear(192 → 64) → Tanh → Linear(64 → vocab_size)

    关键技巧：
        用 torch.roll + 逐步 shift 来实现"滑动窗口"拼接，比逐位置 for 循环高效。
        序列开头之前的"虚空"位置用特殊 <BLANK> token（index = vocab_size）填充。
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # Embedding 表：+1 是因为需要一个额外的 <BLANK> token（index = vocab_size）
        # 用于填充窗口开头"还没有字符"的位置
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # MLP：输入是 block_size 个 embedding 拼在一起
        # block_size * n_embd → n_embd2 → vocab_size
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        """
        滑动窗口拼接的实现方式：

        假设 block_size=3，输入 idx = [a, b, c, d, e]，我们想得到：
            位置 0 的输入：[<B>, <B>, a]  → 预测 b
            位置 1 的输入：[<B>, a,   b]  → 预测 c
            位置 2 的输入：[a,   b,   c]  → 预测 d
            ...

        代码实现：循环 block_size 次，每次把 idx 向右 roll 一位，
        然后把当前 idx 的 embedding 收集起来，最后拼接。

        第 0 次（k=0）：idx=[a,b,c,d,e]    → 收集当前位置 embedding
        第 1 次（k=1）：idx=[<B>,a,b,c,d]  → 收集前 1 位 embedding
        第 2 次（k=2）：idx=[d,<B>,a,b,c]  → 收集前 2 位 embedding
        最后 cat([emb2, emb1, emb0])，得到每个位置"前3个字符"的表示
        """
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)          # (B, T, n_embd)：当前 idx 的 embedding
            idx = torch.roll(idx, 1, 1)      # 整体向右 roll 一位（时间轴方向）
            idx[:, 0] = self.vocab_size      # 最左边空出来的位置填 <BLANK>
            embs.append(tok_emb)

        # 拼接：每个 emb (B, T, n_embd) → 拼后 (B, T, n_embd * block_size)
        x = torch.cat(embs, -1)

        logits = self.mlp(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
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
        row = X_samp[i, 1:].tolist()
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

    parser = argparse.ArgumentParser(description="MLP 语言模型")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt')
    parser.add_argument('--work-dir',   '-o', type=str, default='out_mlp')
    parser.add_argument('--resume',     action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--num-workers', '-n', type=int, default=4)
    parser.add_argument('--max-steps',  type=int, default=-1)
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--seed',       type=int, default=3407)
    parser.add_argument('--top-k',      type=int, default=-1)
    parser.add_argument('--n-embd',  type=int, default=64, help="每个字符 embedding 维度")
    parser.add_argument('--n-embd2', type=int, default=64, help="MLP 隐藏层维度")
    parser.add_argument('--batch-size',    '-b', type=int,   default=32)
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

    # 初始化模型
    config = ModelConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=args.n_embd, n_embd2=args.n_embd2
    )
    model = MLP(config)
    model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")
    print(f"  - Embedding 层: {model.wte.weight.numel()} ({vocab_size+1} tokens × {args.n_embd} dims)")
    print(f"  - MLP 层: {sum(p.numel() for p in model.mlp.parameters())}")

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

        logits, loss = model(X, Y)

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 10 == 0:
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
