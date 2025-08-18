from torch_geometric.datasets import Planetoid
import torch
import my_gcn_cuda  # Your pybind11 module

dataset = Planetoid(root='data', name='Cora')
data = dataset[0].to('cuda')

# Just test reading from feature tensor
my_gcn_cuda.test_read_tensor(data.x)
