# Transformer 英中翻译学习项目

本项目的目标是使用 PyTorch 从零搭建并训练一个 **英文翻译成中文（English → Chinese）** 的 Transformer 模型。从注意力、前馈网络和归一化等基础模块开始，逐步完成 Encoder–Decoder、数据处理、训练与推理，理解完整的机器翻译流程。

目前已实现单层编码器和解码器 block，并在 Notebook 中探索英中平行语料。后续计划自行组装完整模型并从头训练，输入英文句子，输出中文翻译。

当前尚未包含完整的翻译模型、训练流程和文本生成接口。

## 项目结构

| 文件 | 内容 |
| --- | --- |
| `attention.py` | 缩放点积注意力、多头注意力、Q/K/V 投影与输出投影 |
| `feedforward.py` | 两层线性变换与 ReLU 组成的逐位置前馈网络 |
| `layernorm.py` | 手写 LayerNorm |
| `encoder.py` | 自注意力与前馈网络组成的编码器 block |
| `decoder.py` | 因果自注意力、交叉注意力与前馈网络组成的解码器 block |
| `LLMlearn.ipynb` | 加载 OPUS-100 英中数据及 CUDA 初步实验 |

## 环境准备

建议在独立的 Python 虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

若需要运行 Notebook，再安装：

```bash
python -m pip install datasets notebook ipywidgets
jupyter notebook LLMlearn.ipynb
```

基础 block 示例可以在 CPU 上运行。Notebook 中的 `a.cuda(0)` 需要可用的 CUDA 环境；使用 CPU 时可跳过该单元。首次加载数据集需要联网下载数据。

## 运行一个 Encoder–Decoder block

在项目根目录运行以下代码：

```python
import torch

from encoder import EncoderTransformerBlock
from decoder import DecoderTransformerBlock

torch.manual_seed(42)

batch_size = 2
src_len = 7
tgt_len = 5
dim = 32
head_n = 4

encoder = EncoderTransformerBlock(dim, head_n=head_n)
decoder = DecoderTransformerBlock(dim, head_n=head_n)

# 用随机特征演示接口；完整模型中应由词嵌入和位置编码提供输入。
src = torch.randn(batch_size, src_len, dim)
tgt = torch.randn(batch_size, tgt_len, dim)

# True 表示允许关注，保留下三角及对角线，屏蔽未来位置。
causal_mask = torch.ones(tgt_len, tgt_len, dtype=torch.bool).tril()

encoder.eval()
decoder.eval()
with torch.no_grad():
    memory = encoder(src)
    output = decoder(tgt, memory, causal_mask)

print(memory.shape)  # torch.Size([2, 7, 32])
print(output.shape)  # torch.Size([2, 5, 32])
```

该示例展示 block 的连接方式和输出形状，输出仍是特征向量，并非翻译文本。

## 模块约定

- 输入采用 `[批次大小, 序列长度, 特征维度]`，即 `[B, L, D]`。
- `dim` 是特征维度；`head_n` 是正整数，且必须整除 `dim`。
- `dila_w` 是前馈网络的隐藏维度扩展倍数，默认 `4`，对应隐藏维度 `dim * 4`。
- 编码器和解码器均采用 Post-LN：每个子层先计算输出，再与输入残差相加，最后执行 LayerNorm。
- Decoder 交叉注意力的 Q 来自 Decoder 状态，K、V 来自 Encoder 输出。源序列与目标序列可以具有不同长度。

多头注意力的分数形状为 `[B, H, Lq, Lk]`，其中 `H` 为头数。`mask` 必须是可广播到该形状的布尔张量，且与输入位于同一设备：

| 用途 | mask 形状示例 |
| --- | --- |
| 因果自注意力 | `[L, L]` |
| 按批次屏蔽 padding 键位置 | `[B, 1, 1, Lk]` |

本项目约定 `True` 为允许关注，`False` 为屏蔽。每个 query 至少应保留一个可关注位置，整行屏蔽会使 softmax 产生 NaN。

## 计划使用的数据集

训练数据暂定使用 Hugging Face 上的 `Helsinki-NLP/opus-100`，选择 `en-zh` 配置。Notebook 已包含以下加载与查看代码：

```python
from datasets import load_dataset

dataset = load_dataset(
    "Helsinki-NLP/opus-100",
    "en-zh"
)

print(dataset)
print(dataset["train"][:10])
```

翻译方向固定为 **英文作为源文本，中文作为目标文本**。加载后先检查数据划分、字段、样本内容和句长分布，再完成清洗、分词、编码和批次构造。目前数据加载与 block 实现尚未连接成训练流程。

## 后续实现路线

以下内容为计划，尚未全部实现：

1. **准备数据**：检查 OPUS-100 英中样本，处理空文本和过长句子；保留训练、验证和测试划分，先用少量训练样本打通流程。
2. **构建分词与词表**：确定英文和中文的分词方案，仅使用训练集拟合分词器或构建词表；定义 `<pad>`、`<bos>`、`<eos>`，以及分词方案需要时的 `<unk>`，完成文本与 token ID 的转换。
3. **组装完整 Transformer**：补充源语言和目标语言的词嵌入、位置编码、多层 Encoder/Decoder 堆叠，以及映射到中文目标词表的输出投影。
4. **构造训练批次与 mask**：实现变长序列 padding、源序列 padding mask，以及目标序列的因果 mask 和 padding mask，并接入对应注意力层。
5. **实现训练循环**：采用 teacher forcing，将中文目标序列右移作为 Decoder 输入，预测下一个 token；使用忽略 padding 的交叉熵损失，完成反向传播、优化器更新和验证。
6. **检查训练效果**：先确认模型能够拟合一个小训练子集，再逐步扩大数据和模型规模；记录训练与验证损失，保存模型、分词器、词表及训练配置。
7. **实现翻译推理**：编码英文输入，从 `<bos>` 开始逐 token 生成中文，遇到 `<eos>` 或达到长度上限时停止；先实现贪心解码，再按需加入 beam search。
8. **评估翻译质量**：在验证集上调整配置，最终在独立测试集上评估；结合中文分词约定计算 BLEU 等指标，并人工查看翻译样例。

训练时的目标序列对齐示意（以字为 token 举例，实际边界由分词器决定）：

```text
英文输入：       I love you.
Decoder 输入：   <bos> 我 爱 你 。
预测标签：       我 爱 你 。 <eos>
```

最终希望提供一个可保存、可加载的模型，以及接收英文句子并返回中文翻译的推理入口。

## 当前限制与后续工作

- **LayerNorm 尚需与标准公式对齐**：当前偏置初始化为 `1`，使用默认样本标准差，并在标准差外加 `eps`。后续可改为零偏置、总体方差，以及 `sqrt(var + eps)`；特征维度为 `1` 时，当前标准差计算会产生 NaN。
- **Padding mask 接口尚未补齐**：底层多头注意力支持 mask，但 Encoder 的 `forward` 和 Decoder 的交叉注意力调用尚未暴露源序列 mask。
- 尚未加入 dropout。
- 尚未实现分词、词表、词嵌入、位置编码、多层 block 堆叠及词表输出投影。
- 尚未实现目标序列右移、损失函数、优化器、训练循环、推理解码和翻译评估。
- 仓库尚未包含自动化测试；建议后续验证不同源/目标长度、因果屏蔽、padding 屏蔽以及梯度传播。
