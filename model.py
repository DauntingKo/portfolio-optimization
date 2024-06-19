import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
# from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm


class WeightedSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        edge_weight = None
    ) -> Tensor:
        if edge_weight is None:
            raise ValueError("Edge weights are required for weighted aggregation.")
        
        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("WeightedSAGEConv output contains NaN or Inf during forward pass")
        return out

    def message(self, x_j, edge_weight):
        edge_weight = edge_weight.float()
        if edge_weight.numel() > 0:
            min_edge_weight = torch.min(edge_weight)
            max_edge_weight = torch.max(edge_weight)
            if min_edge_weight == max_edge_weight:
                # print(f'min_edge_weight = max_edge_weight = {max_edge_weight}')
                normalized_edge_weight = torch.ones_like(edge_weight) #[1,1,1....]
            else:
                normalized_edge_weight = (edge_weight - min_edge_weight) / (max_edge_weight - min_edge_weight)
            if torch.isnan(edge_weight).any():
                raise ValueError("message edge_weight contains NaN or Inf during forward pass")
            if torch.isnan(normalized_edge_weight).any():
                raise ValueError("message normalized_edge_weight contains NaN or Inf during forward pass")
            # print("x_j size= ",x_j.shape)
            # print("edge_weight size= ",normalized_edge_weight.shape)
            # print("message out= ",(x_j * normalized_edge_weight.view(-1, 1)).shape)
            # 將邊權重應用到鄰居節點
            return x_j * normalized_edge_weight.view(-1, 1)
        else:
            return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

# class WeightedSAGEConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, normalize=False, root_weight=True, bias=True, **kwargs):
#         super(WeightedSAGEConv, self).__init__(node_dim=0, aggr='add', **kwargs)  # 使用'add'聚合
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalize = normalize
#         self.root_weight = root_weight

#         # 目標節點、鄰居節點特徵做線性轉換
#         self.lin_self = torch.nn.Linear(in_channels, out_channels, bias=bias)
#         self.lin_neigh = torch.nn.Linear(in_channels, out_channels, bias=False)

#         # 根節點權重
#         if root_weight:
#             self.root_lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_self.reset_parameters()
#         self.lin_neigh.reset_parameters()
#         if self.root_weight:
#             self.root_lin.reset_parameters()

#     def forward(self, x, edge_index, edge_weight=None):

#         if edge_weight is None:
#             raise ValueError("Edge weights are required for weighted aggregation.")
        
#         # 目標節點、鄰居節點特徵做線性轉換
#         self_x = self.lin_self(x)
#         neigh_x = self.lin_neigh(x)
        
#         # 對鄰居節點加入權重，使用propagate方法聚合信息
#         out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=neigh_x, edge_weight=edge_weight)
        
#         # 如果使用根節點權重，將根節點特徵加入
#         if self.root_weight:
#             out += self.root_lin(self_x)
#         else:
#             out += self_x

#         if self.normalize:
#             out = F.normalize(out, p=2, dim=-1)
            
#         return out

#     def message(self, x_j, edge_weight):
#         edge_weight = edge_weight.float()
#         if edge_weight.numel() > 0:
#             min_edge_weight = torch.min(edge_weight)
#             max_edge_weight = torch.max(edge_weight)
#             normalized_edge_weight = (edge_weight - min_edge_weight) / (max_edge_weight - min_edge_weight)
#             print("x_j size= ",x_j.shape)
#             print("edge_weight size= ",normalized_edge_weight.shape)
#             print("message out= ",(x_j * normalized_edge_weight.view(-1, 1)).shape)
#             # 將邊權重應用到鄰居節點
#             return x_j * normalized_edge_weight.view(-1, 1)
#         else:
#             return x_j

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    
class SAGEWEIGHT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super(SAGEWEIGHT, self).__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList() #batch normalization

        if n_layers == 1:
            self.layers.append(WeightedSAGEConv(in_channels, out_channels, normalize=False))
        elif n_layers == 2:
            self.layers.append(WeightedSAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layers.append(WeightedSAGEConv(hidden_channels, out_channels, normalize=False))
        else:
            self.layers.append(WeightedSAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 2):
            self.layers.append(WeightedSAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))

        if n_layers > 2: 
            self.layers.append(WeightedSAGEConv(hidden_channels, out_channels, normalize=False))

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers
        
        for i, layer in enumerate(looper):
            x = layer(x, edge_index, edge_weight=edge_weight)
            try:
                x = self.layers_bn[i](x)
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=-1), torch.var(x)

    def inference(self, total_loader, device, data):
        xs = []
        var_ = []
        for batch in total_loader:
            batch_edge_weights = data.edge_weights[batch.e_id]
            out, var = self.forward(batch.x.to(device), batch.edge_index.to(device), batch_edge_weights.to(device))
            out = out[:batch.batch_size]
            xs.append(out.cpu())
            var_.append(var.item())

        out_all = torch.cat(xs, dim=0)

        return out_all, var_

#Graph Sample and Aggregation
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super(SAGE, self).__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList() #batch normalization

        if n_layers == 1:
            self.layers.append(SAGEConv(in_channels, out_channels, normalize=False))
        elif n_layers == 2:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        else:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))

        if n_layers > 2: 
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers
        
        for i, layer in enumerate(looper):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=-1), torch.var(x)

    def inference(self, total_loader, device):
        xs = []
        var_ = []
        for batch in total_loader:
            out, var = self.forward(batch.x.to(device), batch.edge_index.to(device))
            out = out[:batch.batch_size]
            xs.append(out.cpu())
            var_.append(var.item())

        out_all = torch.cat(xs, dim=0)

        return out_all, var_
    
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.nll_loss(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            self.alpha = self.alpha.to(input.device)
            focal_loss = self.alpha[target] * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss