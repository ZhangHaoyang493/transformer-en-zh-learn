import torch
import torch.nn as nn

from attention import MultiHeadAttention
from layernorm import LayerNormLayer
from feedforward import FeedForwardNetwork

class EncoderTransformerBlock(nn.Module):
    def __init__(self, dim, head_n=4, dila_w=4):
        super(EncoderTransformerBlock, self).__init__()

        self.feedforward = FeedForwardNetwork(dim, dim*dila_w)
        self.multiHeadAttn = MultiHeadAttention(dim, head_n=head_n)
        self.layernorm = LayerNormLayer(dim)
        self.layernorm2 = LayerNormLayer(dim)

    def forward(self, x):
        x1 = self.multiHeadAttn(x, x, x)
        x1 = self.layernorm(x + x1)
        x2 = self.feedforward(x1)
        x2 = self.layernorm2(x2 + x1)
        return x2


