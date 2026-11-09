import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vovab_size, dim):
        super(EmbeddingLayer).__init__()

        self.vocab_emb = nn.Embedding(vovab_size, dim)

    def forward(self, idx):
        return self.vocab_emb(idx)


class PositionEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(PositionEncoding).__init__()

        self.pos_emb = torch.ones(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        # 2i: sin(pos / 10000^(2i/d))
        # 2i+1: cos(pos / 10000^(2i/d))
        div_term = torch.pow(torch.arange(0, 2, dim) / dim, 10000)
        self.pos_emb[:, 0::2] = torch.sin(position / div_term)
        self.pos_emb[:, 1::2] = torch.cos(position / div_term)

    def forward(self, x):
        return x + self.pos_emb

