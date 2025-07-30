import torch
import graphcuda

print(graphcuda.add_forward(torch.ones(2), torch.ones(2)))