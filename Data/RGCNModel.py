import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter
from typing import Optional, Tuple, Union


class RGCNConv(MessagePassing):    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,     
        aggr: str = 'mean',
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]
        
        if num_bases is not None:
            # Basis Decomposition
            self.weight = Parameter(torch.empty(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))  # relation -> bases
        else:
            self.weight = Parameter(torch.empty(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)
        
        if root_weight:
            self.root = Parameter(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, Tuple[Optional[Tensor], Tensor]],
                edge_index: Tensor, edge_type: Tensor):

        if isinstance(x, Tensor) or x is None:
            x_l = x_r = x
        else:
            x_l, x_r = x[0], x[1]
        
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
            x_r = x_l
        
        size = (x_l.size(0), x_r.size(0))
        
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)
        
        for i in range(self.num_relations):
            mask = (edge_type == i)
            if not mask.any():
                continue
                
            tmp_edge_index = edge_index[:, mask]
            
            if not torch.is_floating_point(x_r):
                h = self.propagate(tmp_edge_index, x=weight[i, x_l], size=size)
            else:
                h = self.propagate(tmp_edge_index, x=x_l, size=size)
                h = h @ weight[i]                     
            
            out = out + h
        
        if self.root is not None:
            if not torch.is_floating_point(x_r):
                out = out + self.root[x_r]
            else:
                out = out + x_r @ self.root
        
        # 3. bias
        if self.bias is not None:
            out = out + self.bias
            
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j