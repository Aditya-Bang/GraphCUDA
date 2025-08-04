#include <torch/extension.h> // Required for PyTorch C++ extensions
#include <torch/script.h>  // General LibTorch header, often useful

/**
 * @brief Performs the forward pass of a Graph Convolutional Network (GCN) layer.
 *
 * This function computes Y = (A @ X) @ W, optionally applying a ReLU activation.
 * It also returns the mask used for ReLU, which is necessary for the backward pass.
 *
 * @param X Input feature matrix (N x in_dim).
 * @param adjm Adjacency matrix (N x N).
 * @param weights Weight matrix for the layer (in_dim x out_dim).
 * @param apply_relu Boolean flag indicating whether to apply ReLU activation.
 * @return A tuple containing:
 * - The output tensor Y (N x out_dim).
 * - The ReLU mask tensor (N x out_dim) if apply_relu is true,
 * otherwise an empty tensor.
 */
std::tuple<torch::Tensor, torch::Tensor> gcn_conv_forward(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu) {

    // Perform the core GCN operation: Y = (A @ X) @ W
    // adjm.mm(X) computes (A @ X)
    // .mm(weights) then computes ((A @ X) @ W)
    torch::Tensor y = adjm.mm(X).mm(weights);

    torch::Tensor mask;
    if (apply_relu) {
        // Create the ReLU mask: elements where y > 0 become 1.0, others 0.0
        mask = y.gt(0).to(torch::kFloat32);
        // Apply ReLU: element-wise multiplication with the mask
        return std::make_tuple(y * mask, mask);
    } else {
        // If ReLU is not applied, the mask is not relevant for backward pass
        // and is returned as an empty tensor, similar to Python's None.
        mask = torch::empty({});
        return std::make_tuple(y, mask);
    }
}

/**
 * @brief Performs the backward pass for a Graph Convolutional Network (GCN) layer.
 *
 * This function computes gradients for the input features (dX) and updates the
 * layer's weights (dW) in-place using gradient descent. It accounts for the
 * ReLU mask if it was applied in the forward pass.
 *
 * @param Y_grad Gradient of the loss with respect to the output of this layer (dL/dY).
 * @param X_cached The input feature matrix X from the forward pass (cached).
 * @param adjm The adjacency matrix A from the forward pass.
 * @param weights A reference to the layer's weight matrix (in_dim x out_dim).
 * This tensor will be updated in-place.
 * @param learning_rate The learning rate for weight updates.
 * @param apply_relu Boolean flag indicating if ReLU was applied in the forward pass.
 * @param mask_cached The ReLU mask generated during the forward pass.
 * @return The gradient of the loss with respect to the input features X (dL/dX).
 */
torch::Tensor gcn_conv_backward(
    torch::Tensor Y_grad,
    torch::Tensor X_cached, // Original input X from forward pass
    torch::Tensor adjm,
    torch::Tensor& weights, // Weights passed by reference for in-place update
    double learning_rate,
    bool apply_relu,
    torch::Tensor mask_cached) { // Mask from forward pass

    // If ReLU was applied in the forward pass, mask the incoming gradient
    // This implements the derivative of ReLU: dL/dZ * (1 if Z > 0 else 0)
    if (apply_relu) {
        Y_grad = Y_grad * mask_cached;
    }

    // Calculate the intermediate term (A @ X)
    // This was computed in the forward pass as well, but we recompute here
    // for clarity and to avoid caching too many large tensors if not strictly necessary.
    torch::Tensor A_X = adjm.mm(X_cached);

    // Calculate the gradient for the weights (dL/dW)
    // dL/dW = (A @ X)^T @ dL/dZ (where dL/dZ is the masked Y_grad)
    torch::Tensor dW = A_X.transpose(0, 1).mm(Y_grad);

    // Calculate the gradient for the input features (dL/dX)
    // dL/dX = A @ (dL/dZ @ W^T)
    torch::Tensor dX = adjm.mm(Y_grad.mm(weights.transpose(0, 1)));

    // Perform the gradient descent step for weights
    // weights -= learning_rate * dW (in-place update)
    weights.sub_(learning_rate * dW);

    // Return the gradient for the input features, to be propagated to previous layers
    return dX;
}

// PYBIND11_MODULE is a macro that creates the entry point for a Python extension module.
// TORCH_EXTENSION_NAME is a macro provided by PyTorch's build system,
// which expands to the name of your module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose the gcn_conv_forward function to Python
    // m.def("python_function_name", &cxx_function_name, "docstring")
    m.def("gcn_conv_forward", &gcn_conv_forward, "GCN Convolution Forward Pass");
    // Expose the gcn_conv_backward function to Python
    m.def("gcn_conv_backward", &gcn_conv_backward, "GCN Convolution Backward Pass");
}
