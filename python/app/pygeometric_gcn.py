import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# Load Cora dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')), name='Cora')
data = dataset[0]  # Cora contains a single graph

# Define GCN Model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Set up model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)

print(f"Model is on device: {next(model.parameters()).device}")
print(f"Data.x is on device: {data.x.device}")
print(f"Data.edge_index is on device: {data.edge_index.device}")

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Test function
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs

# Test before training (epoch 0)
train_acc, val_acc, test_acc = test()
print(f"Epoch 000 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

epochs = 20

torch.cuda.synchronize() if torch.cuda.is_available() else None
start_time = time.time()

for epoch in range(1, epochs + 1):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    epoch_start = time.time()

    loss = train()
    train_acc, val_acc, test_acc = test()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    epoch_time = time.time() - epoch_start

    # if epoch % 1 == 0:
        # print(f"Epoch {epoch:03d} | Time: {epoch_time:.4f}s | Loss: {loss:.4f} | "f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")


# Final sync before total time
torch.cuda.synchronize() if torch.cuda.is_available() else None
total_time = time.time() - start_time

print(f"\nTotal training + testing time for {epochs} epochs: {total_time:.4f} seconds")
print(f"Average time per epoch (train + test): {total_time / epochs:.6f} seconds")


import timeit

# Define a wrapper function for just training once
def single_train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Measure time for 100 runs
time = timeit.timeit(single_train, number=100)
print(f"Average train() time: {time / 100:.6f} seconds")
