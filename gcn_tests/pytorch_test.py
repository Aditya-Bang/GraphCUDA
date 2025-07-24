import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid


# Load dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')), name='Cora')
data = dataset[0]

class ManualGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ManualGCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adjm):
        x = adjm @ x  # A * X
        x = self.linear(x)  # (A * X) * W
        return x

class ManualGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_index, num_nodes):
        super(ManualGCN, self).__init__()

        self.adjm = self._normalize_adj(edge_index, num_nodes)

        self.gcn1 = ManualGCNLayer(input_dim, hidden_dim)
        self.gcn2 = ManualGCNLayer(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        x = self.gcn1(x, self.adjm)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, self.adjm)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def _normalize_adj(edge_index, num_nodes):
        # Step 1: Initialize adjacency matrix A
        A = torch.zeros((num_nodes, num_nodes))
        A[edge_index[0], edge_index[1]] = 1

        # Step 2: Add self-loops: A_hat = A + I
        A += torch.eye(num_nodes)

        # Step 3: Degree matrix D
        degree = A.sum(dim=1)
        D_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        # Step 4: Normalize: D^-1/2 * A * D^-1/2
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
        A_norm = D_mat_inv_sqrt @ A @ D_mat_inv_sqrt

        return A_norm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ManualGCN(dataset.num_node_features, 16, dataset.num_classes, data.edge_index, data.num_nodes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training and testing
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs

# Initial evaluation
train_acc, val_acc, test_acc = test()
print(f"Epoch 000 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

# Training loop
for epoch in range(1, 31):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
