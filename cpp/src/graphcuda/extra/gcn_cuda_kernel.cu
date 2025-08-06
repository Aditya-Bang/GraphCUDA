#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device functions for ReLU and its derivative
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
    return fmax(0.0, z);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_relu(scalar_t z) {
    return z > 0.0 ? 1.0 : 0.0;
}


// CUDA Kernel for element-wise ReLU forward pass and mask generation
template <typename scalar_t>
__global__ void gcn_relu_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> y_accessor,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mask_accessor) {

    // Calculate global index for 2D tensor
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    const int n = blockIdx.y;                             // Row index

    if (n < y_accessor.size(0) && c < y_accessor.size(1)) {
        scalar_t val = y_accessor[n][c];
        mask_accessor[n][c] = d_relu(val); // Store 1.0 if val > 0, 0.0 otherwise
        y_accessor[n][c] = relu(val);      // Apply ReLU in-place
    }
}

// CUDA Kernel for element-wise ReLU backward pass (masking gradient)
template <typename scalar_t>
__global__ void gcn_relu_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_y_accessor,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mask_accessor) {

    // Calculate global index for 2D tensor
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    const int n = blockIdx.y;                             // Row index

    if (n < grad_y_accessor.size(0) && c < grad_y_accessor.size(1)) {
        grad_y_accessor[n][c] = grad_y_accessor[n][c] * mask_accessor[n][c]; // Apply mask in-place
    }
}


// Higher-level CUDA function for GCN forward pass
// This function will be called by the C++ dispatcher.
torch::Tensor gcn_forward_cuda_kernel(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu,
    torch::Tensor& mask_out) { // mask_out is passed by reference to be filled

    // Perform the core GCN operation: Y = (A @ X) @ W
    // LibTorch's .mm() automatically uses CUDA if tensors are on GPU.
    torch::Tensor y = adjm.mm(X).mm(weights);

    if (apply_relu) {
        // Prepare mask_out tensor
        mask_out = torch::empty_like(y);

        // Launch custom ReLU forward kernel
        const int batch_size = y.size(0);
        const int state_size = y.size(1);
        const int threads_per_block = 1024; // Max threads per block
        const dim3 blocks((state_size + threads_per_block - 1) / threads_per_block, batch_size);

        AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "gcn_relu_forward_cuda", ([&] {
            gcn_relu_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                mask_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));
        return y; // y has been modified in-place by the kernel
    } else {
        // If ReLU is not applied, return an empty mask on the same device as y
        mask_out = torch::empty({}, y.options());
        return y;
    }
}

// Higher-level CUDA function for GCN backward pass
// This function will be called by the C++ dispatcher.
std::tuple<torch::Tensor, torch::Tensor> gcn_backward_cuda_kernel(
    torch::Tensor Y_grad,
    torch::Tensor X_cached,
    torch::Tensor adjm,
    const torch::Tensor& weights, // const reference as weights are not updated here
    bool apply_relu,
    torch::Tensor mask_cached) {

    // If ReLU was applied in the forward pass, mask the incoming gradient
    if (apply_relu) {
        // Launch custom ReLU backward kernel
        const int batch_size = Y_grad.size(0);
        const int state_size = Y_grad.size(1);
        const int threads_per_block = 1024;
        const dim3 blocks((state_size + threads_per_block - 1) / threads_per_block, batch_size);

        AT_DISPATCH_FLOATING_TYPES(Y_grad.scalar_type(), "gcn_relu_backward_cuda", ([&] {
            gcn_relu_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                Y_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                mask_cached.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));
    }

    // Calculate the intermediate term (A @ X)
    torch::Tensor A_X = adjm.mm(X_cached);

    // Calculate the gradient for the weights (dL/dW)
    torch::Tensor dW = A_X.transpose(0, 1).mm(Y_grad);

    // Calculate the gradient for the input features (dL/dX)
    torch::Tensor dX = adjm.mm(Y_grad.mm(weights.transpose(0, 1)));

    return std::make_tuple(dX, dW);
}
