import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphcuda import matmul

dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data')), name='Cora')
data = dataset[0]

edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
edge_index, edge_weight = gcn_norm(edge_index, num_nodes=data.num_nodes)
adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
adj = adj.to(device)

# Get node features (X) and adjacency matrix (adj)
X = data.x  # Node features


# Instead of using Cora features
N, D_in, D_out = 2048, 1024, 512  # rows, cols of A and B


indices = torch.randint(0, N, (1, N//100), device=device).repeat(2, 1)
values = torch.randn(indices.size(1), device=device)
A_sparse = torch.sparse_coo_tensor(indices, values, (N, D_in), device=device)
A = A_sparse.to_dense()
# A = torch.randn(N, D_in, device=device)
B = torch.randn(D_in, D_out, device=device)


#### EXTREMELY IMPORTANT, WARMUP effects timings a lot for manual matmul, torch.matmul already warmed up for some reason?
# Warmup
for _ in range(10):
    _ = torch.matmul(A, B)
    _ = matmul(A, B)


# Time GraphCUDA (your C++ wrapper)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
C1 = matmul(A, B)
end.record()
torch.cuda.synchronize()
graphcuda_time = start.elapsed_time(end)

# Time PyTorch native
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
C2 = torch.matmul(A, B)
end.record()
torch.cuda.synchronize()
pytorch_time = start.elapsed_time(end)

print(f"GraphCUDA matmul: {graphcuda_time:.6f} ms")
print(f"PyTorch matmul:   {pytorch_time:.6f} ms")
print(f"Close results? {torch.allclose(C1, C2, atol=1e-4)}")