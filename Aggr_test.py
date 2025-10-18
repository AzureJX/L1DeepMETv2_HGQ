import torch

edge_index = torch.tensor([
    [0, 2, 2, 0, 3],  # source nodes
    [1, 4, 3, 3, 1]   # target nodes
], dtype=torch.long)

row, col = edge_index

edge_feat = torch.tensor([
    [2,5],
    [7,12],
    [8,1],
    [5,0],
    [41,2]
], dtype=torch.long)

N = 4
F = 3
out_dim = 2
E = 5

out = torch.zeros(N, out_dim, dtype=edge_feat.dtype)
row_expanded = row.unsqueeze(1).expand(-1, out_dim)
# out.scatter_reduce_(0, row_expanded, edge_feat, reduce='mean', include_self=False)
out.scatter_add_(0, row_expanded, edge_feat)
print(out)