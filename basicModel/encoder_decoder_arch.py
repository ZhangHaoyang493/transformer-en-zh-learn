import torch
import torch.nn as nn

from basicModel.encoder import EncoderTransformerBlock
from basicModel.decoder import DecoderTransformerBlock

from basicModel.embedding import PositionEncoding, EmbeddingLayer

class EncoderDecoderArch(nn.Module):
    def __init__(self, dim, src_vocab, tar_vocab, enc_n=4, dec_n=4, head_n=4, dila_w=4, max_len=512):
        super(EncoderDecoderArch, self).__init__()

        self.encoderList = nn.ModuleList(
            [
                EncoderTransformerBlock(dim, head_n, dila_w) for _ in range(enc_n)
            ]
        )
        self.decoderList = nn.ModuleList(
            [
                DecoderTransformerBlock(dim, head_n, dila_w) for _ in range(dec_n)
            ]
        )

        self.src_emb = EmbeddingLayer(src_vocab, dim)
        self.tgt_emb = EmbeddingLayer(tar_vocab, dim)

        self.out_linear = nn.Linear(dim, tar_vocab)
        self.posEnc = PositionEncoding(dim, max_len)

    def encode(self, x, mask):
        x = self.posEnc(self.src_emb(x))
        for m in self.encoderList:
            x = m(x, mask)
        return x

    def decode(self, x, x_enc, tar_mask, src_mask):
        x = self.posEnc(self.tgt_emb(x))
        for m in self.decoderList:
            x = m(x, x_enc, tar_mask, src_mask)
        return x

    def out_to_prob(self, x):
        return self.out_linear(x)

    def forward(self, x_enc, x_dec, src_mask, tgt_mask):
        enc_res = self.encode(x_enc, src_mask)
        dec_res = self.decode(x_dec, enc_res, tgt_mask, src_mask)
        out = self.out_linear(dec_res) # B, len, tar_vocab
        return out


if __name__ == '__main__':
    model = EncoderDecoderArch(32, 1000, 1000, 2, 2, 4, 2, 10).cuda(0)
    input = torch.LongTensor([1, 2, 9, 20, 12, 42, 132, 423, 891, 213]).cuda(0).unsqueeze(0) # 1, 10
    print('input dim: ', input.shape)

    src_mask = torch.ones(1, 1, 1, 10).cuda(0) == 1


    res = torch.zeros(1, 1).cuda(0).type_as(input.data)

    enc_fea = model.encode(input, src_mask)

    for i in range(9):
        out = model.decode(
            res,
            enc_fea,
            torch.tril(torch.ones(res.shape[1], res.shape[1]), diagonal=0).cuda(0) == 1,
            src_mask
        )

        out = model.out_linear(out)
        _, next_word_idx = torch.max(out[:, -1], dim=-1)
        print(res.shape, next_word_idx.unsqueeze(0).shape)
        res = torch.concat([res, next_word_idx.unsqueeze(0)], dim=-1)
        print(res)
