from torch_geometric.nn.conv import transformer_conv, GCNConv
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, global_add_pool, SAGEConv, GATConv, SAGPooling, GraphConv, GCN2Conv, GATv2Conv, GeneralConv, PDNConv, GMMConv 
from torch_geometric.nn import global_mean_pool, global_max_pool

class GNN_v2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, use_weights = True):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(64, out_channels)
        self.use_weights = use_weights
    
    def forward(self, x, edge_index, edge_weight, initial_x):
        if self.use_weights:
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight).relu()
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.linear(x)
        return x.log_softmax(dim=-1)
    
class GNN_v1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return x.log_softmax(dim=-1)