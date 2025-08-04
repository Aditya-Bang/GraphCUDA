#include <torch/extension.h>
#include <torch/script.h>

// A custom autograd Function for the GCN convolution layer.
// This is the idiomatic way to implement a custom backward pass in PyTorch.
class GCNConvFunction : public torch::autograd::Function<GCNConvFunction> {
public:
    // The forward pass.
    // It takes the input tensors and returns the output tensor.
    // We also save any tensors needed for the backward pass in the `ctx` context.
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& x,
        const torch::Tensor& adj,
        const torch::Tensor& weight,
        bool apply_relu) {
        
        // Save the tensors needed for the backward pass.
        // We need x, adj, weight, and the output before ReLU to compute gradients correctly.
        ctx->save_for_backward({x, adj, weight});

        // Step 1: Matrix multiplication (adj * x)
        torch::Tensor support = torch::matmul(adj, x);

        // Step 2: Matrix multiplication (support * weight)
        torch::Tensor output = torch::matmul(support, weight);

        if (apply_relu) {
            // Apply ReLU and save the output to compute the gradient of ReLU during backward pass.
            output = torch::relu(output);
            ctx->mark_non_differentiable({adj}); // Adjacency matrix is not differentiable.
        }

        // Store a boolean flag to indicate if ReLU was applied.
        ctx->saved_data["apply_relu"] = apply_relu;
        
        return output;
    }

    // The backward pass.
    // It takes the context and the gradient of the output, and returns the gradients of the inputs.
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        // Retrieve the saved tensors from the context.
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto adj = saved[1];
        auto weight = saved[2];

        // The gradient of the output.
        auto grad_output = grad_outputs[0];

        // If ReLU was applied in the forward pass, we need to handle its gradient.
        if (ctx->saved_data["apply_relu"].to<bool>()) {
            // The gradient of ReLU is 1 for positive inputs and 0 otherwise.
            // We multiply the incoming gradient by this mask.
            // The `~` operator is used for bitwise NOT on the boolean tensor,
            // which is the correct way to invert a boolean tensor in C++.
            grad_output = grad_output.masked_fill(~grad_output.gt(0), 0.0);
        }

        // Compute the gradient with respect to x.
        // grad_x = A_hat^T * (grad_output * W^T)
        // Since A_hat is symmetric, A_hat^T = A_hat
        auto grad_x = torch::matmul(adj, torch::matmul(grad_output, weight.transpose(-2, -1)));

        // Compute the gradient with respect to the weight matrix.
        // grad_weight = (A_hat * X)^T * grad_output
        auto support = torch::matmul(adj, x);
        auto grad_weight = torch::matmul(support.transpose(-2, -1), grad_output);

        // The gradient with respect to the adjacency matrix is usually not needed,
        // but for completeness, we can compute it.
        // grad_adj = grad_output * W^T * X^T
        auto grad_adj = torch::matmul(torch::matmul(grad_output, weight.transpose(-2, -1)), x.transpose(-2, -1));

        // Return the gradients in the same order as the inputs to the forward pass
        // (x, adj, weight, apply_relu). `adj` and `apply_relu` are non-differentiable, so we return a null tensor for them.
        return {grad_x, torch::Tensor(), grad_weight, torch::Tensor()};
    }
};

// A Python-facing wrapper function to call our custom autograd function.
torch::Tensor gcn_conv_forward(const torch::Tensor& x, const torch::Tensor& adj, const torch::Tensor& weight, bool apply_relu) {
    return GCNConvFunction::apply(x, adj, weight, apply_relu);
}

// PYBIND11_MODULE to expose the function to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GNN library with custom autograd function.";

    // Expose the gcn_conv_forward function.
    m.def("gcn_conv_forward", &gcn_conv_forward, "GCN convolution forward pass with custom backward.");
}
