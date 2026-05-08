import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, LayerNorm, GELU, Dropout, ReLU
from torch_geometric.nn import JumpingKnowledge
import sys
sys.path.append('/mnt/GeneGnn/Github/Data') 
from RGCNModel import RGCNConv

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DeathGRN(torch.nn.Module):
    def __init__(self, in_channels=1284, hidden_channels=256, out_channels=2, num_layers=3, num_relations=2):
        super(DeathGRN, self).__init__()
        
        self.proj = Linear(in_channels, hidden_channels)
        self.proj_norm = BatchNorm1d(hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))
            self.norms.append(BatchNorm1d(hidden_channels))

        self.jk = JumpingKnowledge(mode='max')

        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            GELU(),
            Dropout(0.5), 
            Linear(hidden_channels // 2, hidden_channels // 4),
            LayerNorm(hidden_channels // 4),
            GELU(),
            Dropout(0.5), 
            Linear(hidden_channels // 4, out_channels)
        )

    def forward(self, x, edge_index, edge_type):
        x = self.proj(x)
        x = self.proj_norm(F.relu(x))
        x = F.dropout(x, p=0.3, training=self.training)

        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(F.relu(x))
            x = F.dropout(x, p=0.3, training=self.training) 
            xs.append(x)

        x_jk = self.jk(xs)

        logits = self.classifier(x_jk)
        return logits