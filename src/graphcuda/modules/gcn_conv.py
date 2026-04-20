import torch
from typing import Optional

from graphcuda.ops._fused_spmm_gemm_act.triton_impl_small_n import fused_spmm_gemm_relu_small_n
from graphcuda.bsr_rm import create_bsr_values_rm


class GCNConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        apply_relu: bool = True,
    ):
        super().__init__()
        
        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")
        
        # TODO: check what value i should put here
        if out_channels > 128:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"out_channels > 128")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.apply_relu = apply_relu
        
        self._cached_adjm_bsr = None
        self._cached_edge_weight = None
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        pass
