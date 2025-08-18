// my_op.cpp

#include <torch/extension.h>

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_forward);
}
