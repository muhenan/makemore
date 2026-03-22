
# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### 核心概念

#### 词表（Vocabulary）& Tokenizer

词表是"符号 → 整数 index"的映射表。本项目词表只有 27 个 token（26 个小写字母 + 1 个特殊 token）。

Tokenizer 负责 encode（文本 → index）和 decode（index → 文本），在 `CharDataset` 里实现：

```python
self.stoi = {ch: i+1 for i, ch in enumerate(chars)}  # encode
self.itos = {i: s for s, i in self.stoi.items()}      # decode
```

词表和 Tokenizer 是配套的，两者绑定在一起。GPT-4 的词表有 10 万个 token，使用 BPE（字节对编码）把常见字符组合合并成单个 token。

#### Token Embedding（wte）

```python
self.wte = nn.Embedding(vocab_size, n_embd)  # (27, 64)
```

一张可学习的矩阵，把整数 index 映射成 64 维连续向量。整数本身没有语义，embedding 让相似的字符可以有相近的向量，训练时由梯度调整。

"语义相近的词 embedding 接近"不是人为设计的，是训练的副产品：模型为了更好地预测，会把出现在相似上下文里的词推向相近的方向，语义相似性自然涌现。

**业界做法：**
- **从头训练**：大模型训练时 embedding 层作为模型的一部分端到端一起训，无特殊处理
- **用预训练的**：OpenAI 有专门的 `text-embedding` API，输入文本返回向量，用于语义搜索、RAG 等场景；HuggingFace 上也有大量开源预训练 embedding 模型
- **继承权重**：Fine-tune 时直接用 GPT-2、LLaMA 的权重初始化，embedding 层一并继承

小公司基本不从头训 embedding，直接调 API 或用开源预训练权重。

#### Position Embedding

并行模型（BoW、Transformer）同时处理所有位置，不知道每个字符在第几位，需要额外注入位置信息。RNN 不需要，因为它串行处理，天然知道位置。

**原始论文（Vaswani 2017）：三角函数，固定不可学习**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

低维用高频波（相邻位置差异大），高维用低频波（远距离才有差异），每个位置都有唯一的向量指纹。优点是可以外推到训练时没见过的更长序列。

**现代业界主流：RoPE（旋转位置编码）**

LLaMA、Mistral、GPT-NeoX 等主流大模型都在用。不是把位置向量加到 embedding 上，而是把位置信息"旋转"进 Q 和 K 里，天然支持长序列外推，实践效果更好。三角函数方案现在基本只在论文和教学中出现。

#### 三者的完整流程

```
原始文本 "emma"
    ↓ tokenizer
[5, 13, 13, 1]              整数序列
    ↓ wte
(4, 64) 语义向量
    ↓ + wpe
(4, 64) 语义 + 位置          送入模型
```

#### 预训练

大模型预训练的方式和这个项目本质完全一样，就是**预测下一个 token**。

GPT 系列用的预训练目标叫 **CLM（Causal Language Modeling）**：

```
输入：The quick brown fox
目标：quick brown fox jumps
```

每个位置预测下一个 token，和本项目一模一样，只是规模不同：

| | 本项目 | 大模型 |
|--|--|--|
| 词表大小 | 27（字符） | ~10万（BPE） |
| 序列长度 | 16 | 4096 ~ 128K |
| 参数量 | 200K | 7B ~ 700B |
| 训练数据 | 32K 个名字 | 全互联网文本 |

训练数据是大量原始文本，没有任何人工标注，让模型不断预测下一个 token。这叫**自监督学习**，监督信号来自数据本身，不需要人打标签。

BERT 不同，用的是 **MLM（Masked Language Modeling）**：随机遮住句子中间的 token，让模型预测被遮住的部分。能同时看左右两边，但无法做生成任务。GPT 用 CLM 适合生成，BERT 用 MLM 适合理解。

### License

MIT
