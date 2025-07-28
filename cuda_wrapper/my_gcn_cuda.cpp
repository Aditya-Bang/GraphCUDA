#include <torch/extension.h>
#include <vector>
#include <iostream>

void test_read_tensor(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Tensor must be contiguous");

    auto x_ptr = x.data_ptr<float>(); // assuming float32 features

    std::cout << "Tensor size: " << x.sizes() << std::endl;

    // Call CUDA kernel
    my_test_kernel_launcher(x_ptr, x.size(0), x.size(1));
}

void my_test_kernel_launcher(float* x, int rows, int cols);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_read_tensor", &test_read_tensor, "Test CUDA kernel read");
}
