import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphcuda import matmul3

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
N, D_in, D_out = 2000, 2000, 2000  # rows, cols of A and B


indices = torch.randint(0, N, (1, N//100), device=device).repeat(2, 1)
values = torch.randn(indices.size(1), device=device)
A_sparse = torch.sparse_coo_tensor(indices, values, (N, D_in), device=device)
A = A_sparse.to_dense()
B = torch.randn(D_in, D_out, device=device)
# A = adj
# B = data.x
# A_sparse = torch.sparse_coo_tensor(
#     edge_index,
#     edge_weight,
#     (data.num_nodes, data.num_nodes),
#     device=device
# ).coalesce()

# Dimensions
print(f"A (adjacency) shape: {A.shape}")
print(f"B (features) shape: {B.shape}")

# # Sparsity
# if A.is_sparse:
#     num_elements = A.shape[0] * A.shape[1]
#     num_nonzero = A._nnz()  # Number of non-zero elements
#     sparsity = 1.0 - (num_nonzero / num_elements)
#     print(f"A sparsity: {sparsity:.4f} ({num_nonzero} / {num_elements} non-zero)")
# else:
#     num_elements = A.numel()
#     num_nonzero = (A != 0).sum().item()
#     sparsity = 1.0 - (num_nonzero / num_elements)
#     print(f"A sparsity: {sparsity:.4f} ({num_nonzero} / {num_elements} non-zero)")

# # B is dense
# num_elements_b = B.numel()
# num_nonzero_b = (B != 0).sum().item()
# sparsity_b = 1.0 - (num_nonzero_b / num_elements_b)
# print(f"B sparsity: {sparsity_b:.4f} ({num_nonzero_b} / {num_elements_b} non-zero)")

# print(f"edge_index shape: {edge_index.shape}")
# print(f"Number of edges: {edge_index.shape[1]}")

#### EXTREMELY IMPORTANT, WARMUP effects timings a lot for manual matmul, torch.matmul already warmed up for some reason?
# Warmup
for _ in range(10):
    _ = torch.matmul(A, B)
    _ = matmul3(A, B)
    _ = torch.sparse.mm(A_sparse, B)


# Time GraphCUDA (your C++ wrapper)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
C1 = matmul3(A, B)
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

# Time PyTorch sparse
start.record()
C_sparse = torch.sparse.mm(A_sparse, B)
end.record()
torch.cuda.synchronize()
sparse_time = start.elapsed_time(end)

def compare_tensors(a, b, atol=1e-4, rtol=1e-5):
    diff = (a - b).abs()
    print(f"Allclose? {torch.allclose(a, b, atol=atol, rtol=rtol)}")
    print(f"Max abs diff: {diff.max().item()}")
    print(f"Mean abs diff: {diff.mean().item()}")
    rel_diff = diff / (b.abs() + 1e-8)
    print(f"Max rel diff: {rel_diff.max().item()}")
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        mask = ~torch.isclose(a, b, atol=atol, rtol=rtol)
        print("First few mismatches:")
        idx = mask.nonzero(as_tuple=False)
        for i in range(min(20, idx.size(0))):
            r, c = idx[i]
            print(f"[{r}, {c}]: {a[r,c].item()} vs {b[r,c].item()} (diff={diff[r,c].item()})")

print(f"GraphCUDA matmul: {graphcuda_time:.6f} ms")
print(f"PyTorch matmul:   {pytorch_time:.6f} ms")
print(f"PyTorch sparse.mm:     {sparse_time:.6f} ms")
print(f"Close results? {torch.allclose(C1, C2, atol=1e-4)}")
print(f"Sparse vs Dense: {torch.allclose(C_sparse, C2, atol=1e-4)}")

compare_tensors(C1, C2)
