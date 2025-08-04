#include <torch/extension.h>
#include <torch/script.h>

// GCN Convolution Layer
struct GCNConvImpl : torch::nn::Module {
    GCNConvImpl(int in_features, int out_features) {
        weight = register_parameter("weight", torch::randn({in_features, out_features}));
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& adj, bool apply_relu = true) {
        // The GCN convolution formula is: X' = (D^-1/2 * A_hat * D^-1/2) * X * W
        // We will assume a pre-normalized adjacency matrix 'adj' is provided.
        // This corresponds to the `torch_geometric.utils.add_self_loops` and `torch_geometric.utils.normalization.gcn_norm` steps.

        // Step 1: Matrix multiplication (adj * x)
        torch::Tensor support = torch::matmul(adj, x);

        // Step 2: Matrix multiplication (support * weight)
        torch::Tensor output = torch::matmul(support, weight);

        if (apply_relu) {
            output = torch::relu(output);
        }

        return output;
    }

    torch::Tensor weight;
};
TORCH_MODULE(GCNConv);

// GCN Model
struct GCNImpl : torch::nn::Module {
    GCNImpl(int in_features, int hidden_features, int out_features) {
        // Register the layers
        conv1 = GCNConv(in_features, hidden_features);
        conv2 = GCNConv(hidden_features, out_features);
        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& adj) {
        // Pass through the first convolutional layer and apply ReLU
        torch::Tensor x_out = conv1->forward(x, adj, true);

        // Pass through the second convolutional layer
        x_out = conv2->forward(x_out, adj, false);
        
        // Apply log_softmax on the output
        return torch::log_softmax(x_out, /*dim=*/1);
    }

    GCNConv conv1{nullptr}, conv2{nullptr};
};
TORCH_MODULE(GCN);

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// PYBIND11_MODULE to expose the C++ modules to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GNN library implemented in C++ and exposed to Python.";

    m.def("add_forward", &add_forward);

    // Expose the GCNConv layer
    py::class_<GCNConvImpl, std::shared_ptr<GCNConvImpl>>(m, "GCNConv")
        .def(py::init<int, int>())
        .def("forward", &GCNConvImpl::forward);

    // Expose the GCN model
    py::class_<GCNImpl, torch::nn::Module, std::shared_ptr<GCNImpl>>(m, "GCN")
        .def(py::init<int, int, int>())
        .def("forward", &GCNImpl::forward);
}
