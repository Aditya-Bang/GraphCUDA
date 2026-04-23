import math
from typing import Any

import torch
from torch import Tensor


def glorot(value: Any) -> None:
    """Xavier/Glorot uniform init (same scaling as torch_geometric.nn.inits.glorot)."""
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            glorot(v)


def init_gcn_conv_parameters(
    weights: Tensor,
    bias: Tensor | None,
) -> None:
    """Initialize ``GCNConv`` weight with Glorot and bias with zeros (PyG ``GCNConv`` convention)."""
    glorot(weights)
    if bias is not None:
        bias.data.zero_()
