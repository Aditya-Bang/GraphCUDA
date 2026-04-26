import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

try:
    from .. import _graphcuda as cpp_ext
except ImportError as e:
    raise ImportError(
        "Could not import the native C++ extension '_graphcuda'. Original error: " + str(e)
    ) from e


class GCNConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, adjm, weights, apply_relu):
        Y, mask = cpp_ext.gcn_conv_forward(X, adjm, weights, apply_relu)
        ctx.save_for_backward(X, adjm, weights, mask)
        ctx.apply_relu = apply_relu
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X_cached, adjm, weights, mask_cached = ctx.saved_tensors
        apply_relu = ctx.apply_relu
        dX, dW = cpp_ext.gcn_conv_backward(
            grad_Y.contiguous(),
            X_cached,
            adjm,
            weights,
            apply_relu,
            mask_cached
        )
        return dX, None, dW, None

class GCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, apply_relu: bool = True):
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.apply_relu = apply_relu

        stdv = math.sqrt(2.0 / (in_dim + out_dim))
        self.weights = Parameter(torch.Tensor(in_dim, out_dim))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor, adjm: torch.Tensor) -> torch.Tensor:
        return GCNConvFunction.apply(X, adjm, self.weights, self.apply_relu)
