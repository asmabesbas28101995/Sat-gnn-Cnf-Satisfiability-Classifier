import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SATGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 32)
        self.conv2 = GCNConv(32, 32)
        self.lin = nn.Linear(32, 2)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)

        return self.lin(x)
