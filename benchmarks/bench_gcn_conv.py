import os

import torch
from graphcuda.modules.gcn_conv import GCNConv as GraphCUDAGCNConv
from torch_geometric.nn import GCNConv as PyGGCNConv
from torch_geometric.datasets import Planetoid


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

dataset = Planetoid(root=DATA_PATH, name='Cora')
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

gcn_conv_layer = GraphCUDAGCNConv(
    in_channels=dataset.num_node_features,
    out_channels=16,
    improved=False,
    cached=True,
    add_self_loops=True,
    normalize=True,
    bias=True,
    apply_relu=True,
).to(device)

out = gcn_conv_layer(data.x, data.edge_index)
print(out.shape)
