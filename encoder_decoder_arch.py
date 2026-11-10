import torch
import torch.nn as nn

from encoder import EncoderTransformerBlock
from decoder import DecoderTransformerBlock

from embedding import PositionEncoding

class EncoderDecoderArch(nn.Module):
    def __init__(self, dim, enc_n=4, dec_n=4, head_n=4, dila_w=4, max_len=512):
        super(EncoderDecoderArch).__init__()

        self.encoderList = nn.ModuleList(
            [
                EncoderTransformerBlock(dim, head_n, dila_w) for _ in range(enc_n)
            ]
        )
        self.decoderList = nn.ModuleList(
            [
                DecoderTransformerBlock(dim, head_n, dila_w)
            ]
        )

        self.out_linear = nn.Linear(dim, dim)
        self.posEnc = PositionEncoding(dim, max_len)

    def encode(self, x):
        x = self.posEnc(x)
        for m in self.encoderList:
            x = m(x)
        return x

    def decode(self, x, x_enc, mask):
        x = self.posEnc(x)
        for m in self.decoderList:
            x = m(x, x_enc, mask)
        return x

    def forward(self, x):
    