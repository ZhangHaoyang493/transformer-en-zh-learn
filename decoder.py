import torch
import torch.nn as nn

from attention import MultiHeadAttention
from layernorm import LayerNormLayer
from feedforward import FeedForwardNetwork


class DecoderTransformerBlock(nn.Module):
    def __init__(self, dim, head_n=4, dila_w=4):
        super(DecoderTransformerBlock, self).__init__()

        self.feedforward = FeedForwardNetwork(dim, dim*dila_w)
        self.multiHeadAttn = MultiHeadAttention(dim, head_n=head_n)
        self.maskMultiHeadAttn = MultiHeadAttention(dim, head_n=head_n)
        self.layernorm = LayerNormLayer(dim)
        self.layernorm2 = LayerNormLayer(dim)
        self.layernorm3 = LayerNormLayer(dim)

    def forward(self, x, x_enc, tar_mask, src_mask):
        x1 = self.maskMultiHeadAttn(x, x, x, tar_mask)
        x1 = self.layernorm(x + x1)
        x2 = self.multiHeadAttn(x1, x_enc, x_enc, src_mask)
        x2 = self.layernorm2(x1 + x2)
        x3 = self.feedforward(x2)
        x3 = self.layernorm3(x3 + x2)
        return x3