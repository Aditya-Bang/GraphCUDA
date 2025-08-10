#include <torch/extension.h>
#include "gcn.cuh"
#include "matmul.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose GCN conv functions
    m.def("gcn_conv_forward", &gcn_conv_forward, "GCN Convolution Forward Pass");
    m.def("gcn_conv_backward", &gcn_conv_backward, "GCN Convolution Backward Pass");

    // Expose Matmul wrapper function
    m.def("matmul1", &matmul1, "Matrix Multiplication - Global Memory Coalescing");
    m.def("matmul2", &matmul2, "Matrix Multiplication - Shared Memory Cache-Blocking");
    m.def("matmul3", &matmul3, "Matrix Multiplication - 2D Block Tiling and Vectorized Memory Access");
    m.def("matmul4", &matmul4, "Matrix Multiplication - Warptiling");
}
