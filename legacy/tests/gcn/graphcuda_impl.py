import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphcuda import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features, apply_relu=True)
        self.conv2 = GCNConv(hidden_features, out_features, apply_relu=False)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return torch.log_softmax(x, dim=1)


def test_graphcuda_gcn(DATA_PATH: str):
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Planetoid(root=DATA_PATH, name='Cora')
    data = dataset[0]

    data = data.to(device)
    data.x = data.x.to(dtype)
    assert data.x.dim() == 2, "data.x must be 2D"
    
    edge_index, edge_weight = gcn_norm(
        edge_index=data.edge_index,
        edge_weight=None,
        num_nodes=data.num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=dtype,
    )
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]

    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        (data.num_nodes, data.num_nodes),
        device=device
    ).coalesce()

    in_features = dataset.num_node_features
    hidden_features = 16
    out_features = dataset.num_classes
    model = GCN(in_features, hidden_features, out_features).to(device).to(dtype)

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Data.x is on device: {data.x.device}, dtype: {data.x.dtype}")
    print(f"Data.edge_index is on device: {data.edge_index.device}, dtype: {data.edge_index.dtype}")

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, adj)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = model(data.x, adj)
            pred = out.argmax(dim=1)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                correct = pred[mask] == data.y[mask]
                accs.append(int(correct.sum()) / int(mask.sum()))
        return accs

    train_acc, val_acc, test_acc = evaluate()
    print(f"Epoch 000 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    epochs = 10000
    total_time = 0

    def time_pytorch_function(func = None, args = None):
        # CUDA is asynchronous, so we need to synchronize
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        
        model.train()
        optimizer.zero_grad()
        out = model(data.x, adj)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        start.record()
        loss.backward()
        optimizer.step()
        func_output = loss.item()
        end.record()
        torch.cuda.synchronize()
        # Convert milliseconds to seconds
        return start.elapsed_time(end) / 1000, func_output

    # warm-up
    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, adj)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    for epoch in range(1, epochs + 1):
        # epoch_start_time = time.time()
        # loss = train()
        # epoch_end_time = time.time()
        # epoch_time = epoch_end_time - epoch_start_time
        epoch_time, loss = time_pytorch_function(train, None)
        total_time += epoch_time
        train_acc, val_acc, test_acc = evaluate()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | Time: {epoch_time:.4f}s | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    print(f"\nTotal training time for {epochs} epochs: {total_time:.4f} seconds")
    print(f"Average time per epoch (train + test): {total_time / epochs:.6f} seconds")
