"""
Transformer 字符级语言模型
==========================
完全复刻 GPT-2 的结构，论文：
    Vaswani et al. 2017: "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762

和 BoW 的关系：
    BoW 是 Transformer 的简化版，两者骨架完全相同。
    核心升级只有两处：
        1. CausalBoW（等权平均）→ CausalSelfAttention（动态加权）
        2. 单个 Block → 多个 Block 堆叠

结构总览：
    wte               : token embedding
    wpe               : position embedding
    Block × n_layer   : 核心处理单元，每个 Block = Attention + MLP
    ln_f              : 最后的 LayerNorm
    lm_head           : 解码头
"""

import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from makemore import ModelConfig, CharDataset, create_datasets, InfiniteDataLoader, generate

# =============================================================================
# GELU 激活函数
# =============================================================================

class NewGELU(nn.Module):
    """
    GELU（Gaussian Error Linear Unit）激活函数，GPT-2 使用的版本。
    论文：https://arxiv.org/abs/1606.08415

    和 Tanh/ReLU 的区别：
        ReLU:  x < 0 时直接截断为 0（硬截断）
        Tanh:  把值域压到 [-1, 1]
        GELU:  x < 0 时不是硬截断，而是平滑地接近 0，保留一点梯度
               实践中在大模型上比 Tanh 和 ReLU 效果更好

    公式：GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# =============================================================================
# CausalSelfAttention：多头因果自注意力
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    这是 Transformer 和 BoW 的核心区别所在。

    BoW 的做法：att = softmax(全零下三角)  → 等权平均，固定不变
    这里的做法：att = softmax(q @ k.T / sqrt(d))  → 由内容动态决定权重

    Q、K、V 是什么：
        每个位置的输入向量 x 经过三个不同的线性变换，得到三个角色：
        Q（Query）：我想查找什么？
        K（Key）：  我能提供什么信息？
        V（Value）：我实际提供的内容是什么？

        att[i][j] = softmax( Q[i] · K[j] / sqrt(d) )
        含义：位置 i 对位置 j 的关注程度，由 i 的 Query 和 j 的 Key 的相似度决定

    多头（Multi-Head）：
        把 n_embd=64 维拆成 n_head=4 个头，每个头 64/4=16 维。
        每个头独立计算 attention，关注不同类型的模式：
            head1 可能学到：关注相邻字符
            head2 可能学到：关注元音字符
            head3 可能学到：关注词尾模式
            head4 可能学到：...
        最后把 4 个头的输出拼回 64 维。

    为什么除以 sqrt(d)：
        d 是每个头的维度（head_size = n_embd / n_head = 16）
        q @ k 的结果方差约为 d，除以 sqrt(d) 把方差归一化到 1，
        防止点积值太大导致 softmax 梯度消失。

    参数量（n_embd=64）：
        c_attn: 64 × (64×3) + 64×3 = 12,288 + 192 = 12,480  （生成 q/k/v）
        c_proj: 64 × 64 + 64       = 4,096 + 64   = 4,160   （输出投影）
        合计: 16,640
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 确保能整除

        # 一个线性层同时生成 Q、K、V，输出维度是 3 * n_embd
        # 比三个独立的线性层在实现上更高效
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 多头拼接后的输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask：下三角矩阵，shape (1, 1, block_size, block_size)
        # 多了两个维度是为了和 (B, n_head, T, T) 的 att 对齐广播
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # (batch, seq_len, n_embd)

        # ── 第一步：生成 Q、K、V ──────────────────────────────────────────────
        #
        # c_attn 是 nn.Linear(64, 192)，内部有权重矩阵 (192, 64) 和偏置 (192,)
        # 这些权重是模型参数，训练前随机初始化，训练时由梯度更新
        #
        # self.c_attn(x) 等价于：x @ c_attn.weight.T + c_attn.bias
        # x (B, T, 64) → 线性变换 → (B, T, 192)，值是 x 和权重相乘后的浮点数
        #
        # .split(64, dim=2) 把最后一维 192 切成三段，每段 64：
        #   q: (B, T, 64)  ← 输出的 [0:64]   对应权重矩阵的前 64 行
        #   k: (B, T, 64)  ← 输出的 [64:128] 对应权重矩阵的中 64 行
        #   v: (B, T, 64)  ← 输出的 [128:192] 对应权重矩阵的后 64 行
        #
        # q/k/v 已经是 x 和权重相乘之后的结果，不是原始的 x
        # 三份值不同，因为它们来自同一个 192 维输出的不同切片（对应不同权重）
        # 等价于三个独立线性层：q = x @ Wq，k = x @ Wk，v = x @ Wv
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # ── 第二步：拆分多头 ──────────────────────────────────────────────────
        #
        # hs = head_size = 64 // 4 = 16，每个头的维度
        hs = C // self.n_head

        # .view(B, T, 4, 16)：把 shape (B, T, 64) 重新解读为 (B, T, 4, 16)
        #   内存没有任何变化，数据原封不动，只是改变了索引方式
        #   把最后的 64 维看成"4个头，每个头16维"
        #
        # .transpose(1, 2)：交换第1维(T)和第2维(n_head)
        #   (B, T, 4, 16) → (B, 4, T, 16)
        #   这次内存布局会改变（所以后面合并时需要 .contiguous()）
        #
        # 为什么要 transpose：
        #   后续 q @ k.T 做的是最后两维的矩阵乘法
        #   transpose 后最后两维是 (T, 16)，乘出来是 (T, T)，即注意力矩阵
        #   transpose 前最后两维是 (4, 16)，乘出来是 (4, 4)，没有意义
        #   B 和 4 这两维是批量维度，PyTorch 对 4 个头自动并行计算，互不干扰
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, 4, T, 16)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, 4, T, 16)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, 4, T, 16)

        # ── 第三步：计算 attention ────────────────────────────────────────────
        #
        # k.transpose(-2, -1)：把 k 最后两维互换
        #   (B, 4, T, 16) → (B, 4, 16, T)
        #
        # q @ k.T：标准矩阵乘法，B 和 4 是批量维度自动广播
        #   (B, 4, T, 16) @ (B, 4, 16, T) → (B, 4, T, T)
        #
        # 结果 att[b][h][i][j] = q[b][h][i] 和 k[b][h][j] 的点积
        # 表示第 h 个头里，位置 i 对位置 j 的原始相似度分数
        # 这就是 T×T 的位置注意力矩阵，每个元素 [i][j] = 位置 i 对位置 j 的关注程度
        #
        # 为什么需要 Q 和 K 两个，而不是直接用 x 自己点积：
        #   如果直接 x[i] · x[j]，结果是对称的，x[i]·x[j] == x[j]·x[i]
        #   意味着"i 关注 j 的程度"永远等于"j 关注 i 的程度"，不符合语言规律
        #   有了 Q 和 K，Q[i]·K[j] != Q[j]·K[i]，注意力矩阵不再对称
        #   Q = x @ Wq  → "我想找什么"的空间
        #   K = x @ Wk  → "我能提供什么"的空间
        #   Q[i]·K[j] 问的是：位置 i 想找的东西，和位置 j 能提供的东西，匹不匹配？
        #   Wq 和 Wk 是两套独立参数，训练时各自优化
        #
        # 除以 sqrt(hs)：hs=16，sqrt(16)=4
        #   q @ k 的结果方差约为 hs，除以 sqrt(hs) 把方差归一化到 1
        #   防止点积值太大导致 softmax 后梯度消失（概率全压到一个位置）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

        # causal mask：右上角填 -inf，确保每个位置只能看左边
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # (B, 4, T, T)
        # ── 激活值说明 ────────────────────────────────────────────────────────
        # 到这里为止，前向传播产生的激活值（非参数，不会保存到 model.pt）：
        #   q, k, v : (B, 4, T, 16) 各一份
        #   att     : (B, 4, T, T)
        # 这些值训练时必须保留在显存里，反向传播算梯度时需要用到它们。
        # 例如 att 对 q 求梯度时，必须知道当时 att 的值是多少。
        # 反向传播完成后才会释放。
        #
        # 激活值的显存消耗（B=32, T=16, n_head=4, hs=16, float32=4字节）：
        #   att: 32×4×16×16×4 = 131,072 字节 ≈ 128KB，乘以 4 层 Block ≈ 512KB
        # batch size 越大、序列越长，激活值以 T² 速度增长，
        # 训练大模型时显存瓶颈往往是激活值而不是参数本身。

        # ── 第四步：用 V 做加权求和 ───────────────────────────────────────────
        #
        # att @ v：(B, 4, T, T) × (B, 4, T, 16) → (B, 4, T, 16)
        # 和 BoW 的 att @ x 完全一样的操作，只是权重不再均等
        # 每个头独立完成自己的加权求和，4 个头并行，互不干扰
        y = att @ v  # (B, 4, T, 16)

        # ── 第五步：拼回多头，输出投影 ────────────────────────────────────────
        #
        # transpose(1, 2)：(B, 4, T, 16) → (B, T, 4, 16)，把 n_head 移回去
        # .contiguous()：transpose 后内存不连续，view 之前必须先整理内存布局
        # .view(B, T, C)：(B, T, 4, 16) → (B, T, 64)，把 4 个头拼回一个向量
        #
        # 此时 y 是 4 个头的简单拼接：
        #   [head0的16维 | head1的16维 | head2的16维 | head3的16维]
        #   各个头的信息还是相互独立的，head0 的值不受 head1 影响
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # c_proj 是 Linear(64, 64)，让每个输出值都能看到全部 64 维的输入
        # 作用：混合各个头的信息，head0 和 head1 的内容在这里产生交互
        # 没有这一层，4 个头只是拼接，永远不会互相影响
        y = self.c_proj(y)  # (B, T, 64)
        return y

# =============================================================================
# Block：单个 Transformer 块
# =============================================================================

class Block(nn.Module):
    """
    一个完整的 Transformer Block，和 BoWBlock 结构对比：

        BoWBlock:
            x = x + CausalBoW(x)          # 等权平均，无 LayerNorm
            x = x + MLP_tanh(x)           # Tanh 激活，hidden=n_embd2

        Transformer Block:
            x = x + Attention(LayerNorm(x))  # 动态加权，先 LayerNorm
            x = x + MLP_gelu(LayerNorm(x))   # GELU 激活，hidden=4*n_embd

    两处新增：

    1. LayerNorm（层归一化）：
       在每个子层之前对输入做归一化（均值0，方差1），
       稳定训练过程，让梯度更好地流动。
       这是 Pre-Norm 版本（归一化在子层之前），GPT-2 的做法。

    2. MLP hidden 维度是 4 * n_embd：
       BoW 是 n_embd → n_embd2 → n_embd（64→64→64）
       这里是 n_embd → 4*n_embd → n_embd（64→256→64），表达能力更强。

    参数量（n_embd=64）：
        ln_1:   64×2 = 128         （LayerNorm 的 scale 和 bias）
        attn:   16,640             （见 CausalSelfAttention）
        ln_2:   64×2 = 128
        c_fc:   64×256+256 = 16,640
        c_proj: 256×64+64  = 16,448
        合计: ~50,000
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)           # Attention 前的归一化
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)           # MLP 前的归一化
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd),  # 扩张 4 倍
            c_proj = nn.Linear(4 * config.n_embd, config.n_embd),  # 压缩回来
            act    = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x):
        # Pre-Norm：先 LayerNorm，再 Attention，再残差
        x = x + self.attn(self.ln_1(x))
        # Pre-Norm：先 LayerNorm，再 MLP，再残差
        x = x + self.mlpf(self.ln_2(x))
        return x

# =============================================================================
# Transformer：完整语言模型
# =============================================================================

class Transformer(nn.Module):
    """
    完整 Transformer 语言模型，GPT-2 同款结构。

    和 BoW 的完整对比：

        BoW:
            wte + wpe → BoWBlock × 1 → lm_head

        Transformer:
            wte + wpe → Block × n_layer → LayerNorm → lm_head

    多层堆叠的意义：
        第 1 层：学到字符级的局部模式（哪些字符常见相邻）
        第 2 层：在第 1 层输出的基础上，学到更抽象的组合模式
        第 3、4 层：更高层次的语言规律
        层数越多，能学到越抽象的特征，但训练也越难。

    数据流：
        idx (B, T)
          ↓ wte + wpe
        x (B, T, 64)                    字符语义 + 位置信息
          ↓ Block × 4（每个 Block 都做 Attention + MLP）
        x (B, T, 64)                    经过 4 层深度处理
          ↓ LayerNorm
        x (B, T, 64)
          ↓ lm_head
        logits (B, T, vocab_size)

    参数量（n_layer=4, n_embd=64, n_head=4, block_size=16, vocab_size=27）：
        wte:          27 × 64      = 1,728
        wpe:          16 × 64      = 1,024
        Block × 4:    ~50,000 × 4  = ~200,000
        ln_f:         64 × 2       = 128
        lm_head:      64 × 27      = 1,728（注意：无 bias）
        总计: ~200K（默认配置约 200K 参数）
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # n_layer 个 Block
            ln_f = nn.LayerNorm(config.n_embd),  # 最后的归一化，输出前做一次
        ))
        # lm_head 无 bias：GPT-2 的做法，减少参数，实践中影响不大
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("transformer 参数量: %.2fM" % (n_params / 1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)

        # ── 第一步：双重 Embedding 相加 ───────────────────────────────────────
        # 和 BoW 完全一样，并行模型必须手动注入位置信息
        tok_emb = self.transformer.wte(idx)  # (B, T, 64)：字符是什么
        pos_emb = self.transformer.wpe(pos)  # (1, T, 64)：在哪个位置
        x = tok_emb + pos_emb               # (B, T, 64)：两种信息合并到同一向量

        # ── 第二步：逐层通过 Block ────────────────────────────────────────────
        # 每个 Block 内部：
        #   x = x + Attention(LayerNorm(x))   ← 残差让原始 x 直通，不经过 LN
        #   x = x + MLP(LayerNorm(x))         ← 残差让原始 x 直通，不经过 LN
        # 残差那条路永远是直通的，梯度可以沿这条高速公路一路流回最开始
        #
        # 每层 Block 输入输出 shape 相同 (B, T, 64)，可以任意堆叠多少层
        # 第1层：学字符级局部模式
        # 第2层：在第1层基础上学更抽象的组合
        # 第3、4层：更高层次的规律
        for block in self.transformer.h:
            x = block(x)                    # (B, T, 64) → (B, T, 64)

        # ── 第三步：最后的 LayerNorm ──────────────────────────────────────────
        # 最后一个 Block 输出经过残差加法后分布可能偏移，
        # 归一化到均值 0 方差 1，让 lm_head 每次接收到稳定的输入分布
        x = self.transformer.ln_f(x)        # (B, T, 64)

        # ── 第四步：解码 ──────────────────────────────────────────────────────
        # Linear(64 → 27)，无 bias（GPT-2 的做法，少 27 个参数，影响极小）
        # 把每个位置的 64 维向量映射到词表大小，得到下一个字符的 logit 分布
        logits = self.lm_head(x)            # (B, T, 27)

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

    parser = argparse.ArgumentParser(description="Transformer 语言模型（GPT-2 同款）")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt')
    parser.add_argument('--work-dir',   '-o', type=str, default='out_transformer')
    parser.add_argument('--resume',     action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--num-workers', '-n', type=int, default=4)
    parser.add_argument('--max-steps',  type=int, default=-1)
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--seed',       type=int, default=3407)
    parser.add_argument('--top-k',      type=int, default=-1)
    parser.add_argument('--n-layer', type=int, default=4,  help="Block 堆叠层数")
    parser.add_argument('--n-head',  type=int, default=4,  help="attention 头数")
    parser.add_argument('--n-embd',  type=int, default=64, help="embedding 维度")
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
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd
    )
    model = Transformer(config)
    model.to(args.device)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())}")

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
