import os
import sys

try:
    # print("Importing torch...")
    import torch
    # print("torch imported successfully!")
    if sys.platform.startswith('win'):
        torch_lib_path = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'lib')
        # print(f"Adding {torch_lib_path} to DLL search path...")
        os.add_dll_directory(torch_lib_path)
        # print(f"{torch_lib_path} added to DLL search path successfully!")
except ImportError:
    print("Torch not found. Please ensure it's installed in your environment.")
    sys.exit(1)


try:
    # print("Importing _graphcuda extension...")
    from . import _graphcuda as cpp_ext
    # print("_graphcuda imported successfully!")
except ImportError as e:
    print("Failed to import _graphcuda extension.")
    raise ImportError(
        "Could not import the native C++ extension '_graphcuda'. "
        "This typically means PyTorch is not installed, or the extension "
        "failed to compile correctly. Please ensure PyTorch is installed "
        f"and try rebuilding your package. Original error: {e}"
    ) from e


import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class GCNConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, adjm, weights, apply_relu, learning_rate):
        # Call the C++ forward function. It returns (output_Y, relu_mask).
        Y, mask = cpp_ext.gcn_conv_forward(X, adjm, weights, apply_relu)

        # Save necessary tensors and scalar values for the backward pass.
        # X_cached, adjm, weights, mask_cached are needed by cpp_ext.gcn_conv_backward.
        # apply_relu and learning_rate are scalar/boolean and stored directly on ctx.
        ctx.save_for_backward(X, adjm, weights, mask)
        ctx.apply_relu = apply_relu
        ctx.learning_rate = learning_rate

        # The forward pass of a torch.autograd.Function should return only the output tensor(s)
        # that require gradients. The mask is an intermediate value for backward.
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        # Retrieve saved tensors and attributes from ctx.
        X_cached, adjm, weights, mask_cached = ctx.saved_tensors
        apply_relu = ctx.apply_relu
        learning_rate = ctx.learning_rate

        # Call the C++ backward function.
        # Note: grad_Y needs to be contiguous for C++ operations.
        # The C++ function updates 'weights' in-place and returns dX.
        dX = cpp_ext.gcn_conv_backward(
            grad_Y.contiguous(),
            X_cached,
            adjm,
            weights, # This weight tensor is passed by reference to C++ and updated in-place.
            learning_rate,
            apply_relu,
            mask_cached
        )

        # The backward method must return a gradient for each input to the forward method,
        # in the same order. If an input does not require a gradient, return None.
        # Inputs to forward were: X, adjm, weights, apply_relu, learning_rate
        # 1. dX for X
        # 2. None for adjm (assuming it's not trainable)
        # 3. None for weights (because they are updated in-place by the C++ function)
        # 4. None for apply_relu (boolean, no gradient)
        # 5. None for learning_rate (scalar hyperparameter, no gradient)
        return dX, None, None, None, None

class GCNConv(nn.Module):
    """
    A Graph Convolutional Network (GCN) layer implemented using a custom
    autograd Function that interfaces with C++ backend for forward and backward passes.
    """
    def __init__(self, in_dim: int, out_dim: int, apply_relu: bool = True, learning_rate: float = 0.01):
        """
        Initializes the GCNConv layer.

        Args:
            in_dim (int): Dimension of the input features.
            out_dim (int): Dimension of the output features.
            apply_relu (bool): Whether to apply ReLU activation after the convolution.
            learning_rate (float): The learning rate to be used for in-place weight updates
                                   in the C++ backward function. Note: This is typically
                                   handled by an optimizer, but included here to match
                                   the C++ function's signature.
        """
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.apply_relu = apply_relu
        self.learning_rate = learning_rate

        # Initialize weights as a learnable PyTorch Parameter
        # Using Xavier uniform initialization, as in your original Python code.
        stdv = math.sqrt(2.0 / (in_dim + out_dim))
        self.weights = Parameter(torch.Tensor(in_dim, out_dim))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor, adjm: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the GCN layer.

        Args:
            X (torch.Tensor): Input feature matrix (N x in_dim).
            adjm (torch.Tensor): Adjacency matrix (N x N).

        Returns:
            torch.Tensor: Output feature matrix (N x out_dim).
        """
        # Call the custom autograd Function's 'apply' method.
        # This will internally call GCNConvFunction.forward for the forward pass
        # and GCNConvFunction.backward for the backward pass during backpropagation.
        return GCNConvFunction.apply(X, adjm, self.weights, self.apply_relu, self.learning_rate)


class GCN(torch.nn.Module):
    """
    A two-layer GCN model implemented in Python, using the custom
    GCNConv layer.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        # Define the two GCNConv layers.
        # The apply_relu flag should be set during initialization of GCNConv,
        # not passed during the forward call.
        self.conv1 = GCNConv(in_features, hidden_features, apply_relu=True)
        self.conv2 = GCNConv(hidden_features, out_features, apply_relu=False)

    def forward(self, x, adj):
        # Pass through the first convolutional layer with ReLU activation.
        # The apply_relu is already configured in self.conv1's __init__.
        x = self.conv1(x, adj)
        
        # Pass through the second convolutional layer without ReLU.
        # The apply_relu is already configured in self.conv2's __init__.
        x = self.conv2(x, adj)

        # Apply log_softmax to the final output.
        return torch.log_softmax(x, dim=1)


# This `__all__` variable defines the public API of your package.
__all__ = ["GCN"]
