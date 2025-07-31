import os
import time
import torch
from torch_geometric.datasets import Planetoid

# Load Cora dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')), name='Cora')
data = dataset[0]  # Cora contains a single graph
