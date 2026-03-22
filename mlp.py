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

        # Embedding 表：把整数 index 映射成 n_embd 维的连续向量
        # 比 Bigram 直接查行更有表达力：相似的字符可以有相近的向量
        # +1 是因为需要一个额外的 <BLANK> token（index = vocab_size）
        # 用于填充窗口开头"还没有字符"的位置（序列前面的虚空位置）
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # MLP 两层全连接网络：
        #   输入：block_size 个字符的 embedding 拼在一起，shape = block_size * n_embd
        #   隐藏：n_embd2 个神经元，Tanh 激活引入非线性
        #         没有非线性的话多层线性等价于一层线性，学不到字符组合规律
        #   输出：vocab_size 个 logit，表示下一个字符的得分
        #
        # 参数量明细（默认值 block_size=16, n_embd=64, n_embd2=64, vocab_size=27）：
        #
        #   Embedding:
        #     (vocab_size+1) × n_embd = 28 × 64 = 1,792
        #
        #   Linear1（权重 + 偏置）:
        #     (block_size × n_embd) × n_embd2 + n_embd2
        #     = (16 × 64) × 64 + 64
        #     = 1024 × 64 + 64 = 65,536 + 64 = 65,600
        #
        #   Linear2（权重 + 偏置）:
        #     n_embd2 × vocab_size + vocab_size
        #     = 64 × 27 + 27 = 1,728 + 27 = 1,755
        #
        #   总计: 1,792 + 65,600 + 1,755 = 69,147
        #   对比 Bigram: 27 × 27 = 729，MLP 参数量约是 Bigram 的 95 倍
        #
        # 瓶颈在 Linear1：block_size 越大，输入维度越高，参数量增长越快
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(), # 激活函数，引入非线性。如果不引入非线性，多层线性等价于一层线性，学不到字符组合规律
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # ── 滑动窗口拼接 ──────────────────────────────────────────────────────
        #
        # 目标：对序列里每个位置 t，收集它前面 block_size 个字符的 embedding，
        #       拼在一起作为 MLP 的输入。
        #
        # 假设 block_size=3，输入序列 idx = [a, b, c, d, e]
        # 我们希望得到（每行是一个位置的输入，→ 右边是要预测的目标）：
        #
        #   位置0: [<B>, <B>,  a ] → 预测 b
        #   位置1: [<B>,  a,   b ] → 预测 c
        #   位置2: [ a,   b,   c ] → 预测 d
        #   位置3: [ b,   c,   d ] → 预测 e
        #   位置4: [ c,   d,   e ] → 预测 ?
        #
        # ── roll 的工作原理 ───────────────────────────────────────────────────
        #
        # torch.roll(idx, 1, 1) 把序列在时间轴（dim=1）上整体右移一位，
        # 最右边的元素会绕回到最左边：
        #
        #   原始:  [a, b, c, d, e]
        #   roll:  [e, a, b, c, d]  ← e 从末尾绕回到开头（不想要）
        #
        # 所以 roll 之后立刻把第 0 列覆盖成 <BLANK>：
        #   修正:  [<B>, a, b, c, d]  ← 正确，表示"向左看一位"
        #
        # ── 三次循环过程 ──────────────────────────────────────────────────────
        #
        # k=0: idx = [a, b, c, d, e]      → embed → emb0（当前字符）
        #      roll → [e, a, b, c, d]，修正 → [<B>, a, b, c, d]
        #
        # k=1: idx = [<B>, a, b, c, d]    → embed → emb1（前1个字符）
        #      roll → [d, <B>, a, b, c]，修正 → [<B>, <B>, a, b, c]
        #
        # k=2: idx = [<B>, <B>, a, b, c]  → embed → emb2（前2个字符）
        #      roll → ...（后续不再用 idx）
        #
        # cat([emb0, emb1, emb2], dim=-1) 后每个位置的内容：
        #   位置0: [embed(a),   embed(<B>), embed(<B>)]
        #   位置1: [embed(b),   embed(a),   embed(<B>)]
        #   位置2: [embed(c),   embed(b),   embed(a)  ]
        #   位置3: [embed(d),   embed(c),   embed(b)  ]
        #
        # 注意：emb0 是当前字符，emb1 是前1个，emb2 是前2个
        # 拼接顺序是 [当前, 前1, 前2]，和直觉上的 [前2, 前1, 当前] 相反，
        # 但对 MLP 来说无所谓，因为它会学到每个槽位的语义

        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)          # (B, T, n_embd)
            idx = torch.roll(idx, 1, 1)      # 右移一位
            idx[:, 0] = self.vocab_size      # 修正左边界：填 <BLANK>
            embs.append(tok_emb)

        # ── 数据流总览（block_size=16, n_embd=64 为例）────────────────────────
        #
        # idx                              (B, T)           = (32, 16)
        #   ↓ Embedding × block_size 次
        # embs: 16 个 tensor              各 (B, T, 64)
        #   ↓ cat(dim=-1)
        # x                               (B, T, 64*16)    = (32, 16, 1024)
        #   ↓ Linear(1024 → 64)
        #   ↓ Tanh
        # hidden                          (B, T, 64)        = (32, 16, 64)
        #   ↓ Linear(64 → 27)
        # logits                          (B, T, vocab_size) = (32, 16, 27)

        x = torch.cat(embs, -1)   # (B, T, n_embd * block_size)
        logits = self.mlp(x)      # (B, T, vocab_size)

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
