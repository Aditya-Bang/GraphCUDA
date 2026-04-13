import torch

# TODO: maybe make subclass of torch.tensor or torch.nn.Module, where i have the 3 indexes as block descriptors.
# see grouped matmul example triton
# for actually implementing the fused spmm-gemm-relu see fused attention implementation in triton.
class ModifiedBlockELLPackedSparseMatrix:
    """
    Attributes:
        block_size: int
        block_descriptors: list of tuples (num_nonzero_cols, nonzero_nonzero_col_offsets_array_start_index, block_offsets_start_index)
        nonzero_col_offsets: list of column offsets for non-zero elements in the blocks
        block_offsets: list of non-zero values in the blocks (row-major order)

        e.g. for sparse matrix
        [[0, 0, 0, 3],
         [1, 0, 0, 2],
         [0, 4, 0, 0],
         [5, 0, 0, 6]]
        
        with block m size 2, block descriptors would be:
        [2, 0, 0, 3, 2, 4]

        can be viewed as:
        [(2, 0, 0), # block 1 has 2 non-zero columns, starts at index 0 in nonzero_col_offsets and block_offsets
         (3, 2, 4)] # block 2 has 3 non-zero columns, starts at index 2 in nonzero_col_offsets and index 4 in block_offsets

        nonzero_col_offsets would be:
        [0, 3, 0, 1, 3]

        block_offsets would
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
        SPARSE_M, SPARSE_N = sparse_matrix.shape
        
        self.block_size: int = self.choose_block_m(sparse_matrix)
        self.block_descriptors = torch.tensor([], dtype=torch.int32)
        self.nonzero_col_offsets = torch.tensor([], dtype=torch.int32)
        self.block_offsets = torch.tensor([], dtype=sparse_matrix.dtype)

        for pid_m in range(0, SPARSE_M, self.block_size):
            # TODO
            pass
            


    def get_element(self, i: int, j: int):
        pass

    @staticmethod
    def choose_block_m(sparse_matrix: torch.Tensor) -> int:
        """
        Choose the optimal block size for the given sparse matrix density and size.
        """
        # TODO: Implement a heuristic to choose block size based on matrix dimensions and density
        return 16
