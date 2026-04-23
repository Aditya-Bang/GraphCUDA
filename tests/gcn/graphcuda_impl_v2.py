import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from graphcuda.modules.gcn_conv import GCNConv


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=True, add_self_loops=True, bias=True, apply_relu=True)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=True, add_self_loops=True, bias=True, apply_relu=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def test_pygeometric_gcn(DATA_PATH: str):
    dtype = torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Planetoid(root=DATA_PATH, name='Cora')
    data = dataset[0]

    data = data.to(device)
    data.x = data.x.to(dtype)
    assert data.x.dim() == 2, "data.x must be 2D"
    num_node_features = data.x.shape[1]
    num_node_features_pad = (num_node_features + 15) // 16 * 16
    if num_node_features_pad > num_node_features:
        data.x = F.pad(data.x, (0, num_node_features_pad - num_node_features))

    model = GCN(num_node_features_pad, 16, dataset.num_classes).to(device).to(dtype)

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Data.x is on device: {data.x.device}, dtype: {data.x.dtype}")
    print(f"Data.edge_index is on device: {data.edge_index.device}, dtype: {data.edge_index.dtype}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)


    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate():
        model.eval()
        out = model(data)
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
        # epoch_start = time.time()
        # loss = train()
        # epoch_time = time.time() - epoch_start
        epoch_time, loss = time_pytorch_function(train, None)
        total_time += epoch_time

        train_acc, val_acc, test_acc = evaluate()

        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d} | Time: {epoch_time:.4f}s | Loss: {loss:.4f} | "f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")


    print(f"\nTotal training time for {epochs} epochs: {total_time:.4f} seconds")
    print(f"Average time per epoch (train + test): {total_time / epochs:.6f} seconds")
