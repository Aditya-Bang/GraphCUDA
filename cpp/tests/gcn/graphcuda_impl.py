import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# Import the C++-defined model
from graphcuda import GCN

# Load Cora dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data')), name='Cora')
data = dataset[0]

# --- Pre-processing for the C++ GCNConv ---
# Our C++ GCNConv expects a pre-normalized, dense adjacency matrix.
# We will perform this step here.
edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
edge_index, edge_weight = gcn_norm(edge_index, num_nodes=data.num_nodes)
adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]

# Move data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")
data = data.to(device)
adj = adj.to(device)

# Instantiate the C++ GCN model
in_features = dataset.num_node_features
hidden_features = 16
out_features = dataset.num_classes
model = GCN(in_features, hidden_features, out_features).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
model.train()
for epoch in range(20):
    optimizer.zero_grad()
    # The GCN class is directly from C++
    out = model.forward(data.x, adj)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    out = model.forward(data.x, adj)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')
