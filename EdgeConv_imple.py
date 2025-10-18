import torch
import torch.nn as nn
from hgq.layers import QDense


class EdgeConv(nn.Module):
    """
    Args:
        in_channels (int): Number of input features per node
        out_channels (int): Number of output features per node
        aggr (str): Aggregation method ('max', 'mean', 'sum', 'add')
    """
    
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(EdgeConv, self).__init__()
        self.aggr = aggr
        
        # Dense MLP with depth=1
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, batch=None):
        """        
        Args:
            x: [N, F] - node features for all graphs, F = in_channels
            edge_index: [2, E] - edge indices for all graphs
            batch: [N] - batch assignment for each node (optional)
            
        Returns:
            out: [N, out_channels] - output node features
        """
        src, tar = edge_index
        
        # Collect node features for source and target
        x_i = x[src] # [E, F]
        # x_i = torch.gather(x, 0, src.unsqueeze(1).expand(-1, x.size(1)))
        x_j = x[tar] # [E, F]
        # x_j = torch.gather(x, 0, tar.unsqueeze(1).expand(-1, x.size(1)))
        
        # Create edge features
        edge_feat = torch.cat([x_i, x_j - x_i], dim=1)  # [E, 2F]
        
        # Apply MLP
        edge_feat = self.mlp(edge_feat)  # shape becomes [E, out_channels]
        
        # Aggregate using scatter
        N = x.size(0)
        out = torch.zeros(N, edge_feat.size(1), device=x.device, dtype=x.dtype)
        # shape [N, out]
        
        # Expand src indices for scatter operations
        src_expanded = src.unsqueeze(1).expand(-1, edge_feat.size(1)) # shape [E, out]
        # Aggregate over outcoming edges for each node
        if self.aggr == 'max': # take maximum when multiple edges go to same node
            out.scatter_reduce_(0, src_expanded, edge_feat, reduce='amax', include_self=False)
        elif self.aggr == 'mean':
            out.scatter_reduce_(0, src_expanded, edge_feat, reduce='mean', include_self=False)
        elif self.aggr in ['sum', 'add']:
            out.scatter_add_(0, src_expanded, edge_feat)
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggr}")
        
        return out