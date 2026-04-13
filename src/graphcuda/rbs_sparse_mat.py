import torch


class RBPSparseMat(torch.nn.Module):
    """
    Row Block Packed Sparse Matrix. Matrix that blocks by row dimension and only stores columns that are non-zero in each block.
    
    Inputs:
        sparse_matrix: torch.Tensor. Must be a sparse COO tensor 2d matrix.
    
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
        super().__init__()
        assert sparse_matrix.dim() == 2, "Input matrix must be 2D"
        assert sparse_matrix.is_sparse, "Input matrix must be a sparse tensor"
        SPARSE_M, SPARSE_N = sparse_matrix.shape
        sparse_matrix = sparse_matrix.coalesce()
        rows = sparse_matrix.indices()[0]
        cols = sparse_matrix.indices()[1]
        vals = sparse_matrix.values()

        self.block_size: int = self.choose_block_m(sparse_matrix)
        self._num_rows = SPARSE_M
        self._num_cols = SPARSE_N
        device = sparse_matrix.device

        descriptors: list[int] = []
        nonzero_col_list: list[int] = []
        block_offsets_list: list = []

        for block_idx_m in range(0, SPARSE_M, self.block_size):
            r0 = block_idx_m
            h = min(self.block_size, SPARSE_M - r0)
            mask = (rows >= r0) & (rows < r0 + h)
            start_nc = len(nonzero_col_list)
            start_bo = len(block_offsets_list)
            if not mask.any():
                descriptors.extend([0, start_nc, start_bo])
                continue
            br = rows[mask] - r0
            bc = cols[mask]
            bv = vals[mask]
            uc = torch.unique(bc)
            num_nz = int(uc.numel())
            nonzero_col_list.extend(uc.tolist())
            descriptors.extend([num_nz, start_nc, start_bo])
            idx_map = torch.full((SPARSE_N,), -1, dtype=torch.long, device=device)
            idx_map[uc] = torch.arange(num_nz, device=device, dtype=torch.long)
            k_idx = idx_map[bc]
            sub = torch.zeros((h, num_nz), dtype=sparse_matrix.dtype, device=device)
            sub[br, k_idx] = bv
            block_offsets_list.extend(sub.flatten().tolist())

        self.register_buffer(
            "block_descriptors",
            torch.tensor(descriptors, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "nonzero_col_offsets",
            torch.tensor(nonzero_col_list, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "block_offsets",
            torch.tensor(block_offsets_list, dtype=sparse_matrix.dtype, device=device),
        )

    @property
    def device(self) -> torch.device:
        return self.block_offsets.device

    @property
    def dtype(self) -> torch.dtype:
        return self.block_offsets.dtype

    def get_element(self, i: int, j: int):
        if i < 0 or j < 0 or i >= self._num_rows or j >= self._num_cols:
            return 0.0
        block_idx = i // self.block_size
        local_row = i % self.block_size
        n_desc = self.block_descriptors.numel() // 3
        if block_idx >= n_desc:
            return 0.0
        base = block_idx * 3
        num_nz = int(self.block_descriptors[base].item())
        start_nc = int(self.block_descriptors[base + 1].item())
        start_bo = int(self.block_descriptors[base + 2].item())
        if num_nz == 0:
            return 0.0
        cols = self.nonzero_col_offsets[start_nc : start_nc + num_nz]
        match = (cols == j).nonzero(as_tuple=True)[0]
        if match.numel() == 0:
            return 0.0
        k = int(match[0].item())
        idx = start_bo + local_row * num_nz + k
        return float(self.block_offsets[idx].item())

    @staticmethod
    def choose_block_m(sparse_matrix: torch.Tensor) -> int:
        """
        Choose the optimal block size for the given sparse matrix density and size.
        """
        # TODO: Implement a heuristic to choose block size based on matrix dimensions and density
        return 16
