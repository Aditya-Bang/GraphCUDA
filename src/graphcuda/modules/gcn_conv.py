import torch
from typing import Optional

from graphcuda.utils.bsr_rm import to_sparse_bsr_rm
from graphcuda.utils.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from graphcuda.ops._fused_spmm_gemm_act.triton_impl_small_n import fused_spmm_gemm_relu_small_n
# from graphcuda.ops._fused_spmm_gemm_act.torch_impl_bwd import spmm_gemm_relu_backward_torch_impl
from graphcuda.ops._fused_spmm_gemm_act.torch_impl_bwd import spmm_gemm_relu_backward_pyg_csr_impl


class _GCNConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        adjm_bsr_rm: torch.Tensor,
        adj_t_csr: torch.Tensor,
        X: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
        apply_relu: bool,
    ) -> torch.Tensor:
        Y, relu_mask = fused_spmm_gemm_relu_small_n(adjm_bsr_rm, X, weights, bias, apply_relu)
        ctx.save_for_backward(adj_t_csr, X, weights, bias, relu_mask)
        ctx.apply_relu = apply_relu
        return Y
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_Y: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        adj_t_csr, X, weights, bias, relu_mask = ctx.saved_tensors
        apply_relu = ctx.apply_relu
        dX, dW, dBias = spmm_gemm_relu_backward_pyg_csr_impl(grad_Y, adj_t_csr, X, weights, bias, relu_mask, apply_relu)
        return None, None, dX, dW, dBias, None


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
        
        self._cached_adjm_bsr_rm = None
        self._cached_adj_t_csr = None
        
        self.weights = torch.nn.Parameter(torch.empty(in_channels, out_channels))
        glorot(self.weights)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if edge_weight is not None: # TODO: support edge_weight
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"edge_weight yet.")
        
        # copied preprocessing from torch_geometric.nn.conv.gcn_conv.GCNConv.forward
        if self._cached_adjm_bsr_rm is None:
            if self.normalize:
                edge_index, edge_weight = gcn_norm(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=x.size(-2), # TODO: add self.node_dim param
                    improved=self.improved,
                    add_self_loops=self.add_self_loops,
                    flow="source_to_target", # TODO: add self.flow param
                    dtype=x.dtype
                )
            # from edge_index, create adjm_bsr_rm and adjm_csr
            dense_adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=x.size(-2))[0]
            adjm_bsr_rm = to_sparse_bsr_rm(dense_adj)
            adj_t_csr = dense_adj.t().to_sparse_csr()

            if self.cached:
                self._cached_adjm_bsr_rm = adjm_bsr_rm
                self._cached_adj_t_csr = adj_t_csr
        else:
            adjm_bsr_rm = self._cached_adjm_bsr_rm
            adj_t_csr = self._cached_adj_t_csr

        out = _GCNConvFunction.apply(adjm_bsr_rm, adj_t_csr, x, self.weights, self.bias, self.apply_relu)

        return out
