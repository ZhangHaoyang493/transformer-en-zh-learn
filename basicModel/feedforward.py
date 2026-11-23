import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForwardNetwork, self).__init__()

        self.linear_in = nn.Linear(dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.linear_out(torch.relu(self.linear_in(x)))