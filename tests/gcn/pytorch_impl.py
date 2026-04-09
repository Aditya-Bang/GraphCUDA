import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class ManualGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ManualGCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adjm):
        x = adjm @ x  # A * X
        x = self.linear(x)  # (A * X) * W
        return x

class ManualGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adjm):
        super(ManualGCN, self).__init__()

        self.adjm = adjm

        self.gcn1 = ManualGCNLayer(input_dim, hidden_dim)
        self.gcn2 = ManualGCNLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = self.gcn1(x, self.adjm)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.gcn2(x, self.adjm)
        return F.log_softmax(x, dim=1)

def test_pytorch_manual_gcn(DATA_PATH: str):
    dataset = Planetoid(root=DATA_PATH, name='Cora')
    data = dataset[0]

    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    edge_index, edge_weight = gcn_norm(edge_index, num_nodes=data.num_nodes)
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    adj = adj.to(device)

    in_features = dataset.num_node_features
    hidden_features = 16
    out_features = dataset.num_classes
    model = ManualGCN(in_features, hidden_features, out_features, adj).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Data.x is on device: {data.x.device}")
    print(f"Data.edge_index is on device: {data.edge_index.device}")

    def train():
        model.train()
        optimizer.zero_grad()
        out = model.forward(data.x)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = model.forward(data.x)
            pred = out.argmax(dim=1)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                correct = pred[mask] == data.y[mask]
                accs.append(int(correct.sum()) / int(mask.sum()))
        return accs

    train_acc, val_acc, test_acc = evaluate()
    print(f"Epoch 000 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    epochs = 20
    total_time = 0

    def time_pytorch_function(func, args):
        # CUDA is asynchronous, so we need to synchronize
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        func_output = func(args) if args is not None else func()
        end.record()
        torch.cuda.synchronize()
        # Convert milliseconds to seconds
        return start.elapsed_time(end) / 1000, func_output

    # warm-up
    for _ in range(20):
        train()

    for epoch in range(1, epochs + 1):
        # epoch_start_time = time.time()
        # loss = train()
        # epoch_end_time = time.time()
        # epoch_time = epoch_end_time - epoch_start_time
        epoch_time, loss = time_pytorch_function(train, None)
        total_time += epoch_time
        train_acc, val_acc, test_acc = evaluate()
        print(f"Epoch {epoch:03d} | Time: {epoch_time:.4f}s | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    print(f"\nTotal training + testing time for {epochs} epochs: {total_time:.4f} seconds")
    print(f"Average time per epoch (train + test): {total_time / epochs:.6f} seconds")
