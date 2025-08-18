import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphcuda import GCNConv


dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')), name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features, apply_relu=True)
        self.conv2 = GCNConv(hidden_features, out_features, apply_relu=False)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return torch.log_softmax(x, dim=1)


edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
edge_index, edge_weight = gcn_norm(edge_index, num_nodes=data.num_nodes)
adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
adj = torch.sparse_coo_tensor(
    edge_index,
    edge_weight,
    (data.num_nodes, data.num_nodes),
    device=device
).coalesce()

in_features = dataset.num_node_features
hidden_features = 16
out_features = dataset.num_classes
model = GCN(in_features, hidden_features, out_features).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1)

print(f"Model is on device: {next(model.parameters()).device}")
print(f"Data.x is on device: {data.x.device}")
print(f"Data.edge_index is on device: {data.edge_index.device}")

def train():
    model.train()
    optimizer.zero_grad()
    out = model.forward(data.x, adj)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model.forward(data.x, adj)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = pred[mask] == data.y[mask]
            accs.append(int(correct.sum()) / int(mask.sum()))
    return accs

train_acc, val_acc, test_acc = test()
print(f"Epoch 000 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

epochs = 20
total_time = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    loss = train()
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    total_time += epoch_time
    train_acc, val_acc, test_acc = test()
    print(f"Epoch {epoch:03d} | Time: {epoch_time:.4f}s | Loss: {loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

print(f"\nTotal training + testing time for {epochs} epochs: {total_time:.4f} seconds")
print(f"Average time per epoch (train + test): {total_time / epochs:.6f} seconds")
