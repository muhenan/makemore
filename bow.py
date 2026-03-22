"""
BoW（Bag of Words）字符级语言模型
===================================
BoW 是 RNN 到 Transformer 之间的过渡模型，作者在注释里调侃说它
"looks suspiciously like a CausalAttention module"（看起来可疑地像 Attention）。

核心思想：
    不像 RNN 串行维护隐藏状态，BoW 直接对当前位置之前的所有 token
    的 embedding 做**等权平均**，用平均结果来预测下一个字符。

    位置 t 的表示 = mean( embed(x0), embed(x1), ..., embed(xt) )

和其他模型的对比：
    - RNN：      串行，用隐藏状态压缩历史，有梯度消失问题
    - BoW：      并行，对历史做等权平均，简单但丢失位置信息
    - Transformer：并行，对历史做**加权**平均（权重由内容决定），最强

结构：
    wte          : token embedding（字符 → 向量）
    wpe          : position embedding（位置 → 向量，BoW 需要额外告诉模型位置）
    BoWBlock     : CausalBoW（等权平均）+ 残差连接 + MLP + 残差连接
    lm_head      : 解码头（向量 → logit）
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

from makemore import ModelConfig, CharDataset, create_datasets, InfiniteDataLoader, generate

# =============================================================================
# CausalBoW：因果等权平均
# =============================================================================

class CausalBoW(nn.Module):
    """
    Causal Bag of Words：对每个位置，把它之前（含自身）的所有 token 的向量等权平均。

    "Causal"的意思：只能看左边，不能看右边（未来），保持自回归性质。

    实现方式：用下三角矩阵做 mask，然后 softmax，得到等权平均的注意力矩阵。

    具体步骤（T=4 为例）：

    1. 初始化全零的注意力矩阵 att (B, T, T)：
       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]

    2. 用下三角 mask 把右上角填成 -inf（遮住未来）：
       [[0,   -inf, -inf, -inf],
        [0,   0,    -inf, -inf],
        [0,   0,    0,    -inf],
        [0,   0,    0,    0   ]]

    3. softmax（-inf → 0，剩下的等权）：
       [[1,    0,    0,    0   ],   ← 位置0只看自己
        [0.5,  0.5,  0,    0   ],   ← 位置1平均前2个
        [0.33, 0.33, 0.33, 0   ],   ← 位置2平均前3个
        [0.25, 0.25, 0.25, 0.25]]   ← 位置3平均前4个

    4. att @ x：加权求和，得到每个位置的"历史平均向量"

    注意：这里 att 里所有非 -inf 的值都是 0，softmax 后自动变成等权。
    这和 Transformer 的区别就是：Transformer 的 att 是由 q@k 计算出来的，
    不同位置有不同权重；BoW 的权重永远均等。
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # 下三角矩阵作为 causal mask，注册为 buffer（不参与训练，但跟随模型保存/移动）
        # shape: (1, block_size, block_size)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # (batch, seq_len, n_embd)

        # 初始化全零 att，右上角填 -inf，softmax 后得到等权下三角矩阵
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # (B, T, T)，每行是一个归一化的平均权重

        # 矩阵乘法：(B, T, T) × (B, T, C) → (B, T, C)
        # T 是位置数量，下三角矩阵就是位置信息
        # C 是embedding维度，可以是 64,128,256 都可以，和位置没关系
        # 每个位置的输出 = 它之前所有位置的 embedding 的等权平均
        y = att @ x
        return y

# =============================================================================
# BoWBlock：BoW + 残差连接 + MLP
# =============================================================================

class BoWBlock(nn.Module):
    """
    一个完整的 BoW 处理块，结构和 Transformer Block 几乎一样：

        x = x + CausalBoW(x)    # 聚合历史信息，残差连接保留原始信息
        x = x + MLP(x)          # 对每个位置做非线性变换，残差连接

    残差连接（x = x + ...）的作用：
        让梯度可以直接跳过这一层流回去，避免梯度消失，
        同时保留原始 embedding 的信息，不被覆写。

    对比 Transformer Block：
        Transformer: x = x + Attention(LayerNorm(x))  → 用加权平均代替等权平均
        BoW:         x = x + CausalBoW(x)             → 等权平均，没有 LayerNorm
    """

    def __init__(self, config):
        super().__init__()
        self.cbow = CausalBoW(config)
        # MLP：n_embd → n_embd2 → n_embd，输入输出维度相同，方便残差连接
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(config.n_embd, config.n_embd2),
            c_proj = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)   # 聚合历史，残差
        x = x + self.mlpf(x)   # 非线性变换，残差
        return x

# =============================================================================
# BoW：完整语言模型
# =============================================================================

class BoW(nn.Module):
    """
    完整 BoW 语言模型。

    相比 RNN 多了 position embedding（wpe）：
        RNN 天然有位置感，因为它按顺序处理，第 i 步就知道自己是第 i 步。
        BoW/Transformer 是并行处理的，所有位置同时计算，
        模型本身不知道"我现在在哪个位置"，需要额外加入位置编码。

    数据流：
        idx (B, T)
          ↓ wte                       tok_emb (B, T, n_embd)
          ↓ wpe                       pos_emb (1, T, n_embd)
          ↓ tok_emb + pos_emb         x (B, T, n_embd)        # 字符信息 + 位置信息
          ↓ BoWBlock（CausalBoW + MLP）x (B, T, n_embd)        # 聚合历史 + 变换
          ↓ lm_head                   logits (B, T, vocab_size)

    参数量（n_embd=64, n_embd2=64, block_size=16, vocab_size=27）：
        wte:    27 × 64        = 1,728
        wpe:    16 × 64        = 1,024   ← RNN 没有这个
        CausalBoW: 无可学习参数  = 0       ← 纯计算，权重固定均等
        MLP c_fc:   64×64+64   = 4,160
        MLP c_proj: 64×64+64   = 4,160
        lm_head:    64×27+27   = 1,755
        总计: ~12,827
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # token embedding：字符 → 向量
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding：位置编号 → 向量（并行模型需要显式告知位置）
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.context_block = BoWBlock(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # ── 完整数据流 ────────────────────────────────────────────────────────
        #
        # 输入：
        #   idx (B, T) = (32, 16)   32个名字，每个长度16（含<START>）
        #   Y   (B, T) = (32, 16)   目标：每个位置的下一个字符
        #
        # ── 第一步：双重 Embedding ─────────────────────────────────────────────
        #
        #   tok_emb = wte(idx)   (32, 16, 64)  ← 字符是什么
        #   pos_emb = wpe(pos)   ( 1, 16, 64)  ← 字符在哪个位置
        #   x = tok_emb + pos_emb (32, 16, 64) ← 两种信息直接相加，合并到同一个向量
        #
        #   RNN 不需要 wpe，因为它按顺序处理，天然知道位置。
        #   BoW 并行处理所有位置，模型本身不知道"我在第几步"，必须手动注入位置信息。
        #
        # ── 第二步：BoWBlock ───────────────────────────────────────────────────
        #
        #   x = context_block(x)  (32, 16, 64) → (32, 16, 64)，shape 不变
        #
        #   内部分两个子步骤：
        #
        #   2a. CausalBoW（等权平均）+ 残差连接：
        #
        #       att (32, 16, 16)  下三角等权矩阵，softmax 后长这样（T=4举例）：
        #           位置0: [1,    0,    0,    0   ]  ← 只看自己
        #           位置1: [0.5,  0.5,  0,    0   ]  ← 平均前2个
        #           位置2: [0.33, 0.33, 0.33, 0   ]  ← 平均前3个
        #           位置3: [0.25, 0.25, 0.25, 0.25]  ← 平均前4个
        #
        #       y = att @ x   (32,16,16) × (32,16,64) → (32,16,64)
        #                     每个位置的输出 = 它之前所有位置向量的等权平均
        #
        #       x = x + y     残差连接：原始向量 + 历史平均
        #                     残差的作用：梯度可以直接跳过 BoW 层流回去，
        #                     同时保留原始 embedding 的信息不被覆写
        #
        #   2b. MLP + 残差连接：
        #
        #       x → Linear(64→64) → Tanh → Linear(64→64)
        #       x = x + mlp(x)    对每个位置独立做非线性变换，增强表达能力
        #                         没有这一步，BoW 只能做线性操作，学不到复杂规律
        #
        # ── 第三步：解码 ───────────────────────────────────────────────────────
        #
        #   logits = lm_head(x)  (32, 16, 64) → (32, 16, 27)
        #   loss   = cross_entropy(logits, Y)
        #
        # ── 和其他模型聚合历史的方式对比 ─────────────────────────────────────
        #
        #   Bigram：     不聚合，只看1个字符          并行 ✓  位置编码 ✗
        #   MLP：        拼接固定窗口的原始 embedding  并行 ✓  位置编码 ✗（roll隐含）
        #   RNN：        串行隐藏状态压缩历史          并行 ✗  位置编码 ✗（顺序天然含位置）
        #   BoW：        等权平均所有历史              并行 ✓  位置编码 ✓ 需要
        #   Transformer：加权平均所有历史              并行 ✓  位置编码 ✓ 需要

        b, t = idx.size()
        # 生成位置编号 [0, 1, 2, ..., t-1]，shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx)  # (B, T, n_embd)：字符语义
        pos_emb = self.wpe(pos)  # (1, T, n_embd)：位置信息
        x = tok_emb + pos_emb    # (B, T, n_embd)

        x = self.context_block(x)  # (B, T, n_embd)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

# =============================================================================
# 依赖全局变量的辅助函数
# =============================================================================

def print_samples(num=10):
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
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
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

    parser = argparse.ArgumentParser(description="BoW 语言模型")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt')
    parser.add_argument('--work-dir',   '-o', type=str, default='out_bow')
    parser.add_argument('--resume',     action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--num-workers', '-n', type=int, default=4)
    parser.add_argument('--max-steps',  type=int, default=-1)
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--seed',       type=int, default=3407)
    parser.add_argument('--top-k',      type=int, default=-1)
    parser.add_argument('--n-embd',  type=int, default=64)
    parser.add_argument('--n-embd2', type=int, default=64)
    parser.add_argument('--batch-size',    '-b', type=int,   default=32)
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4)
    parser.add_argument('--weight-decay',  '-w', type=float, default=0.01)
    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"vocab_size={vocab_size}, block_size={block_size}")

    config = ModelConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=args.n_embd, n_embd2=args.n_embd2
    )
    model = BoW(config)
    model.to(args.device)

    total = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total}")
    print(f"  - wte  (token embedding):    {model.wte.weight.numel()}")
    print(f"  - wpe  (position embedding): {model.wpe.weight.numel()}")
    print(f"  - CausalBoW: 0（无可学习参数）")
    print(f"  - MLP:       {sum(p.numel() for p in model.context_block.mlp.parameters())}")
    print(f"  - lm_head:   {sum(p.numel() for p in model.lm_head.parameters())}")

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
