import torch
import torch.nn as nn

def attention(query, key, value, mask=None):
    # query: B len dim
    # key: B len dim
    # value: B len dim
    d_k = query.shape[-1]
    # attn_weight: B len len
    attn_weight = torch.matmul(query, torch.transpose(key, -1, -2)) / (d_k ** 0.5)
    if mask is not None:
        attn_weight = attn_weight.masked_fill(~mask, float('-inf'))
    attn_weight = torch.softmax(attn_weight, -1)
    return torch.matmul(attn_weight, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_n):
        super(MultiHeadAttention, self).__init__()

        assert dim % head_n == 0, 'Dim fault!'
        self.dim = dim
        self.head_n = head_n


        self.q_transform_layer = nn.Linear(self.dim, self.dim)
        self.k_transform_layer = nn.Linear(self.dim, self.dim)
        self.v_transform_layer = nn.Linear(self.dim, self.dim)
        self.out_transformer_layer = nn.Linear(self.dim, self.dim)


    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.q_transform_layer(q)
        k = self.k_transform_layer(k)
        v = self.v_transform_layer(v)

        head_dim = self.dim // self.head_n

        q = q.view(batch_size, -1, self.head_n, head_dim)
        k = k.view(batch_size, -1, self.head_n, head_dim)
        v = v.view(batch_size, -1, self.head_n, head_dim)

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        res = attention(q, k, v, mask) # B, head_n, len, head_dim
        res = torch.transpose(res, 1, 2).contiguous().view(batch_size, -1, self.dim)

        return self.out_transformer_layer(res)
        
