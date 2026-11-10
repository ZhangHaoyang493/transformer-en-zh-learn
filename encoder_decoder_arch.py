import torch
import torch.nn as nn

from encoder import EncoderTransformerBlock
from decoder import DecoderTransformerBlock

from embedding import PositionEncoding, EmbeddingLayer

class EncoderDecoderArch(nn.Module):
    def __init__(self, dim, src_vocab, tar_vocab, enc_n=4, dec_n=4, head_n=4, dila_w=4, max_len=512):
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

        self.src_emb = EmbeddingLayer(src_vocab, dim)

        self.out_linear = nn.Linear(dim, tar_vocab)
        self.posEnc = PositionEncoding(dim, max_len)

    def encode(self, x, mask):
        x = self.posEnc(x)
        for m in self.encoderList:
            x = m(x, mask)
        return x

    def decode(self, x, x_enc, mask):
        x = self.posEnc(x)
        for m in self.decoderList:
            x = m(x, x_enc, mask)
        return x

    def forward(self, x, src_mask, tgt_mask):
        # 输入->embedding
        x = self.src_emb(x)

        enc = self.encode(x, src_mask)
        dec = self.decode(x, enc, tgt_mask)
        out = self.out_linear(dec) # B, len, voc

        