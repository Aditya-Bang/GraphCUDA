import argparse
import copy
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from graphcuda.modules.gcn_conv import GCNConv as GraphCUDAGCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv as PyGGCNConv


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


class PyGGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = PyGGCNConv(input_dim, hidden_dim, cached=True, add_self_loops=True, bias=True)
        self.conv2 = PyGGCNConv(hidden_dim, output_dim, cached=True, add_self_loops=True, bias=True)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


class GraphCUDAGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GraphCUDAGCNConv(input_dim, hidden_dim, cached=True, add_self_loops=True, bias=True, apply_relu=True)
        self.conv2 = GraphCUDAGCNConv(hidden_dim, output_dim, cached=True, add_self_loops=True, bias=True, apply_relu=False)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


@dataclass
class BenchResult:
    name: str
    total_time_s: float
    avg_epoch_time_s: float
    final_loss: float
    train_acc: float
    val_acc: float
    test_acc: float


def parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def prepare_data(data, device: torch.device, dtype: torch.dtype, pad_features: bool):
    data = copy.deepcopy(data).to(device)
    data.x = data.x.to(dtype=dtype)

    input_dim = data.x.shape[1]
    if pad_features:
        padded_input_dim = (input_dim + 15) // 16 * 16
        if padded_input_dim > input_dim:
            data.x = F.pad(data.x, (0, padded_input_dim - input_dim))
        input_dim = padded_input_dim

    return data, input_dim


def train_epoch(model: nn.Module, data, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def evaluate(model: nn.Module, data) -> tuple[float, float, float]:
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs[0], accs[1], accs[2]


def benchmark_model(
    name: str,
    model: nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    num_warm_epochs: int,
    num_training_epochs: int,
    device: torch.device,
) -> BenchResult:
    print(f"\nBenchmarking {name}")

    for _ in range(num_warm_epochs):
        train_epoch(model, data, optimizer)
    torch.cuda.synchronize(device)

    total_time_s = 0.0
    final_loss = float("nan")
    for _ in range(num_training_epochs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss = train_epoch(model, data, optimizer)
        end.record()

        torch.cuda.synchronize(device)
        total_time_s += start.elapsed_time(end) / 1000.0
        final_loss = float(loss.detach().cpu())

    train_acc, val_acc, test_acc = evaluate(model, data)
    torch.cuda.synchronize(device)

    avg_epoch_time_s = total_time_s / num_training_epochs
    result = BenchResult(
        name=name,
        total_time_s=total_time_s,
        avg_epoch_time_s=avg_epoch_time_s,
        final_loss=final_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
    )
    print_result(result)
    return result


def print_result(result: BenchResult):
    print(
        f"{result.name}: total={result.total_time_s:.6f}s, "
        f"avg_epoch={result.avg_epoch_time_s * 1000:.3f}ms, "
        f"loss={result.final_loss:.4f}, "
        f"train_acc={result.train_acc:.4f}, "
        f"val_acc={result.val_acc:.4f}, "
        f"test_acc={result.test_acc:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark two-layer GCN training on Cora.")
    parser.add_argument("--num-warm-epochs", type=int, default=1000)
    parser.add_argument("--num-training-epochs", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_training_epochs <= 0:
        raise ValueError("--num-training-epochs must be positive")
    if args.num_warm_epochs < 0:
        raise ValueError("--num-warm-epochs must be non-negative")
    if args.hidden_dim <= 0 or args.hidden_dim > 128:
        raise ValueError("--hidden-dim must be in [1, 128] for GraphCUDA GCNConv")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA because GraphCUDA GCNConv uses CUDA kernels.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)

    dataset = Planetoid(root=DATA_PATH, name="Cora")
    base_data = dataset[0]

    pyg_data, pyg_input_dim = prepare_data(base_data, device, dtype, pad_features=False)
    graphcuda_data, graphcuda_input_dim = prepare_data(base_data, device, dtype, pad_features=True)

    print(
        f"Cora: nodes={base_data.num_nodes}, edges={base_data.edge_index.shape[1]}, "
        f"features={dataset.num_node_features}, classes={dataset.num_classes}, dtype={args.dtype}"
    )
    print(
        f"Epochs: warmup={args.num_warm_epochs}, measured={args.num_training_epochs}, "
        f"hidden_dim={args.hidden_dim}"
    )

    torch.manual_seed(args.seed)
    pyg_model = PyGGCN(pyg_input_dim, args.hidden_dim, dataset.num_classes).to(device=device, dtype=dtype)
    pyg_optimizer = torch.optim.SGD(pyg_model.parameters(), lr=args.lr)
    pyg_result = benchmark_model(
        "pygeometric_gcn",
        pyg_model,
        pyg_data,
        pyg_optimizer,
        args.num_warm_epochs,
        args.num_training_epochs,
        device,
    )

    torch.manual_seed(args.seed)
    graphcuda_model = GraphCUDAGCN(
        graphcuda_input_dim,
        args.hidden_dim,
        dataset.num_classes,
    ).to(device=device, dtype=dtype)
    graphcuda_optimizer = torch.optim.SGD(graphcuda_model.parameters(), lr=args.lr)
    graphcuda_result = benchmark_model(
        "graphcuda_gcn",
        graphcuda_model,
        graphcuda_data,
        graphcuda_optimizer,
        args.num_warm_epochs,
        args.num_training_epochs,
        device,
    )

    speedup = pyg_result.avg_epoch_time_s / graphcuda_result.avg_epoch_time_s
    print(f"\nGraphCUDA speedup vs PyG: {speedup:.2f}x")


if __name__ == "__main__":
    main()
