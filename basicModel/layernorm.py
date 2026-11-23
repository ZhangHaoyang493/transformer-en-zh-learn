import torch
import torch.nn as nn

class LayerNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNormLayer, self).__init__()

        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.ones(dim))


        self.eps = eps

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        return self.alpha * (x-mu) / (std+self.eps) + self.beta