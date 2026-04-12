import torch

# TODO: maybe make subclass of torch.tensor or torch.nn.Module, where i have the 3 indexes as block descriptors.
# see grouped matmul example triton
# for actually implementing the fused spmm-gemm-relu see fused attention implementation in triton.
class ModifiedBlockELLPackedSparseMatrix:
    """
    Attributes:
        block_size: int
        block_descriptors: list of tuples (num_nonzero_cols, nonzero_col_indices_array_start_index, ell_block_start_index)
        col_indices: list of column indices for non-zero elements in the blocks
        values: list of non-zero values in the blocks (row-major order)

        e.g. for sparse matrix
        [[0, 0, 0, 3],
         [1, 0, 0, 2],
         [0, 4, 0, 0],
         [5, 0, 0, 6]]
        
        with block size 2x2, block descriptors would be:
        [2, 0, 0, 3, 2, 4]

        can be viewed as:
        [(2, 0, 0), # block 1 has 2 non-zero columns, starts at index 0 in col_indices and values
         (3, 2, 4)] # block 2 has 3 non-zero columns, starts at index 2 in col_indices and index 4 in values

        col_indices would be:
        [0, 3, 0, 1, 3]

        values would
        [0, 3, 1, 2, 0, 4, 0, 5, 0, 6]

        can be viewed as

        ell block 1:
        [[0, 3],
         [1, 2]]
        
        ell block 2:
        [[0, 4, 0],
         [5, 0, 6]]

        Notice memory usage is O(non-zero elements * block m)

    """
    def __init__(self, sparse_matrix: torch.Tensor):
        assert sparse_matrix.dim() == 2, "Input matrix must be 2D"
        assert sparse_matrix.is_sparse, "Input matrix must be a sparse tensor"
        
        self.block_size: int = self.choose_block_m(sparse_matrix)

    def get_element(self, i: int, j: int):
        pass

    @staticmethod
    def choose_block_m(sparse_matrix: torch.Tensor) -> int:
        """
        Choose the optimal block size for the given sparse matrix density and size.
        """
        # TODO: Implement a heuristic to choose block size based on matrix dimensions and density
        return 16
