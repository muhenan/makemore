"""
RNN / GRU 字符级语言模型
=========================
实现了两种循环神经网络：

1. Vanilla RNN（基础版）
   - 每一步把"当前输入 + 上一步隐藏状态"压缩成新的隐藏状态
   - 简单但有梯度消失问题，难以记住很久以前的信息

2. GRU（Gated Recurrent Unit，门控循环单元）
   - Kyunghyun Cho et al. 2014: https://arxiv.org/abs/1409.1259
   - 在 RNN 基础上加了两个"门"来控制信息的保留和更新
   - 缓解了梯度消失，能记住更长距离的依赖关系

对比前两个模型的核心进步：
    - Bigram：只看前 1 个字符（无上下文）
    - MLP：  看前固定 block_size 个字符（固定窗口）
    - RNN：  用隐藏状态 h 压缩整个历史，理论上上下文无限长

核心思想：
    每个时间步不再只看一个固定窗口，而是维护一个"记忆向量" h，
    把过去所有信息都压缩进去，每步根据新输入更新这个记忆。
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
# RNNCell：单个时间步的计算单元
# =============================================================================

class RNNCell(nn.Module):
    """
    Vanilla RNN Cell，每个时间步做一件事：
        输入：xt（当前字符的 embedding）+ hprev（上一步的隐藏状态）
        输出：ht（当前步的新隐藏状态）

    公式：
        ht = Tanh( Linear([xt, hprev]) )

    图示（单步）：
        hprev ──┐
                ├─→ cat → Linear → Tanh → ht
        xt    ──┘

    ht 同时扮演两个角色：
        1. 传给下一个时间步作为 hprev（记忆延续）
        2. 传给 lm_head 解码成下一个字符的 logit（当前步的预测）

    参数量：
        Linear: (n_embd + n_embd2) × n_embd2 + n_embd2
               = (64 + 64) × 64 + 64 = 8,256
    """

    def __init__(self, config):
        super().__init__()
        # 输入维度 = 字符 embedding(n_embd) + 隐藏状态(n_embd2)
        # 输出维度 = 新的隐藏状态(n_embd2)
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # 输入两个向量：
        #   xt    : (B, n_embd)  当前字符的 embedding，表示"现在看到了什么"
        #   hprev : (B, n_embd2) 上一步的隐藏状态，  表示"过去记住了什么"
        #
        # 输出一个向量：
        #   ht    : (B, n_embd2) 新的隐藏状态，      表示"综合过去和现在后，现在记住什么"
        #
        # 内部数据流：
        #   xt    (B, 64) ──┐
        #                    ├─→ cat → (B, 128) → Linear(128→64) → Tanh → ht (B, 64)
        #   hprev (B, 64) ──┘
        #
        #   把两个 64 维向量拼成 128 维，过一个线性层压回 64 维，
        #   再用 Tanh 把值域压到 [-1, 1]
        #
        # ht 的双重用途（在 RNN.forward 的循环里）：
        #   hprev = ht          → 向右传：成为下一个 Cell 的输入（记忆延续）
        #   hiddens.append(ht)  → 向下传：进 lm_head 解码成 logit（当前步预测）
        #
        # 同一个向量，两条路。RNN 本质上是同一个 Cell 在时间轴上反复调用，
        # 不管序列多长，参数永远只有这一个 Linear(128→64)。
        #
        # 和 MLP 的本质区别：
        #   MLP：直接拼接固定窗口的原始 embedding → [embed(a), embed(b), embed(c)] → 预测 d
        #   RNN：用压缩后的历史状态预测            → h2（已压缩了 a、b 的信息）→ 预测 d
        #
        #   理论上 h 能记住任意长的历史，但 Vanilla RNN 因为梯度消失，
        #   久远的信息会逐渐被"冲淡"，这就是 GRU 要解决的问题。

        # 公式：ht = Tanh( Linear([xt, hprev]) )
        xh = torch.cat([xt, hprev], dim=1)  # (B, n_embd + n_embd2)
        ht = F.tanh(self.xh_to_h(xh))       # (B, n_embd2)
        return ht

# =============================================================================
# GRUCell：带门控的 RNN Cell
# =============================================================================

class GRUCell(nn.Module):
    """
    GRU Cell，比 Vanilla RNN 多了两个门：

    1. Reset Gate（重置门）r：
       决定"上一步的隐藏状态有多少需要被遗忘"
       r 接近 0 → 忽略过去，重新开始
       r 接近 1 → 完整保留过去的记忆

    2. Update Gate（更新门）z：
       决定"新计算出的候选状态 hbar 有多少替换进 h"
       z 接近 0 → 保持旧状态不变（跳过这一步更新）
       z 接近 1 → 完全用新候选状态替换

    完整公式：
        xh   = cat([xt, hprev])
        r    = Sigmoid( Linear_r(xh) )          # 重置门，(B, n_embd2)，值域 (0,1)
        hbar = Tanh( Linear_hbar([xt, r*hprev]) ) # 候选新状态，融合了被重置后的历史
        z    = Sigmoid( Linear_z(xh) )          # 更新门，(B, n_embd2)，值域 (0,1)
        ht   = (1-z) * hprev + z * hbar         # 插值：旧状态和候选状态的加权混合

    核心优势：
        最后一行 ht = (1-z)*hprev + z*hbar 是一个加权平均，
        梯度可以通过 (1-z)*hprev 这条路直接流回去，
        不会像 Vanilla RNN 一样在长序列中指数级衰减（梯度消失）

    参数量（每个 Linear 都是 (n_embd+n_embd2) × n_embd2 + n_embd2）：
        Linear_z:    128 × 64 + 64 = 8,256
        Linear_r:    128 × 64 + 64 = 8,256
        Linear_hbar: 128 × 64 + 64 = 8,256
        合计: 24,768（是 Vanilla RNN Cell 的 3 倍）
    """

    def __init__(self, config):
        super().__init__()
        # 三个线性层，输入维度都是 n_embd + n_embd2，输出都是 n_embd2
        self.xh_to_z    = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)  # 更新门
        self.xh_to_r    = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)  # 重置门
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)  # 候选状态

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)          # (B, n_embd + n_embd2)

        # 第一步：重置门决定要遗忘多少过去
        r = F.sigmoid(self.xh_to_r(xh))             # (B, n_embd2)，值域 (0, 1)
        hprev_reset = r * hprev                      # 按位相乘，部分或全部清零过去的状态

        # 第二步：用被重置后的历史计算候选新状态
        xhr  = torch.cat([xt, hprev_reset], dim=1)   # (B, n_embd + n_embd2)
        hbar = F.tanh(self.xh_to_hbar(xhr))          # (B, n_embd2)，候选新状态

        # 第三步：更新门决定候选状态有多少比例写入隐藏状态
        z  = F.sigmoid(self.xh_to_z(xh))            # (B, n_embd2)，值域 (0, 1)

        # 第四步：插值混合旧状态和候选状态
        # z=0 → ht = hprev（完全保留旧状态，跳过更新）
        # z=1 → ht = hbar （完全用新候选替换）
        ht = (1 - z) * hprev + z * hbar              # (B, n_embd2)
        return ht

# =============================================================================
# RNN 模型：把 Cell 组装成完整的语言模型
# =============================================================================

class RNN(nn.Module):
    """
    完整的 RNN 语言模型，支持 Vanilla RNN 和 GRU 两种 cell。

    结构：
        wte    : Embedding 表，把字符 index 变成向量
        start  : 可学习的初始隐藏状态 h0（不是固定的全零）
        cell   : RNNCell 或 GRUCell，每步更新隐藏状态
        lm_head: 把隐藏状态解码成下一个字符的 logit

    数据流（以 T=5, n_embd=64, n_embd2=64 为例）：

        idx (B, T)
          ↓ wte
        emb (B, T, 64)          # 所有字符先一次性 embed

        h0 = start (B, 64)      # 初始隐藏状态
          ↓ 逐步循环 T 次
        h1 = cell(emb[:,0,:], h0)   # 看第 1 个字符，更新记忆
        h2 = cell(emb[:,1,:], h1)   # 看第 2 个字符，更新记忆
        h3 = cell(emb[:,2,:], h2)
        ...
        hT = cell(emb[:,T-1,:], h_{T-1})

        hiddens = stack([h1,h2,...,hT])  # (B, T, 64)
          ↓ lm_head
        logits (B, T, vocab_size)        # 每步预测下一个字符

    关键点：
        h 是整个历史的压缩表示，理论上能记住任意长距离的信息。
        实践中 Vanilla RNN 梯度消失严重，GRU 通过门控机制缓解这个问题。

    参数量（Vanilla RNN，n_embd=64, n_embd2=64, vocab_size=27）：
        start:   1 × 64 = 64
        wte:     27 × 64 = 1,728
        RNNCell: (64+64) × 64 + 64 = 8,256
        lm_head: 64 × 27 + 27 = 1,755
        合计: ~11,803

    参数量（GRU，其余相同）：
        GRUCell: 8,256 × 3 = 24,768
        合计: ~28,315
    """

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # 可学习的初始隐藏状态：shape (1, n_embd2)
        # 用 nn.Parameter 而不是固定的全零，让模型自己学一个好的起点
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))

        # Embedding 表（注意：不需要 +1 的 <BLANK> token，RNN 不需要滑动窗口）
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 根据 cell_type 选择使用哪种循环单元
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)

        # 解码头：把隐藏状态映射到词表大小
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # 先把所有字符一次性 embed，比在循环里逐步 embed 更高效
        emb = self.wte(idx)  # (B, T, n_embd)

        # 初始隐藏状态：把 (1, n_embd2) 扩展成 (B, n_embd2)，batch 里每个样本共享同一个起点
        hprev = self.start.expand((b, -1))

        # 核心循环：逐时间步更新隐藏状态
        # 每步只处理一个字符，但隐藏状态 h 携带了所有历史信息
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]          # 取第 i 步的字符 embedding，(B, n_embd)
            ht = self.cell(xt, hprev)  # 更新隐藏状态，(B, n_embd2)
            hprev = ht                 # 当前隐藏状态传给下一步
            hiddens.append(ht)

        # 把 T 步的隐藏状态堆叠成 (B, T, n_embd2)
        hidden = torch.stack(hiddens, 1)

        # 解码：每个时间步的隐藏状态 → 下一个字符的 logit
        logits = self.lm_head(hidden)  # (B, T, vocab_size)

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

    parser = argparse.ArgumentParser(description="RNN / GRU 语言模型")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt')
    parser.add_argument('--work-dir',   '-o', type=str, default='out_rnn')
    parser.add_argument('--resume',     action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--num-workers', '-n', type=int, default=4)
    parser.add_argument('--max-steps',  type=int, default=-1)
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--seed',       type=int, default=3407)
    parser.add_argument('--top-k',      type=int, default=-1)
    parser.add_argument('--cell-type',  type=str, default='rnn', help="rnn 或 gru")
    parser.add_argument('--n-embd',     type=int, default=64, help="字符 embedding 维度")
    parser.add_argument('--n-embd2',    type=int, default=64, help="隐藏状态维度")
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
    model = RNN(config, cell_type=args.cell_type)
    model.to(args.device)

    total = sum(p.numel() for p in model.parameters())
    print(f"模型类型: {args.cell_type.upper()}")
    print(f"模型总参数量: {total}")
    print(f"  - start (初始隐藏状态): {model.start.numel()}")
    print(f"  - wte   (Embedding):   {model.wte.weight.numel()}")
    print(f"  - cell  ({args.cell_type.upper()}Cell):      {sum(p.numel() for p in model.cell.parameters())}")
    print(f"  - lm_head (解码头):    {sum(p.numel() for p in model.lm_head.parameters())}")

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
