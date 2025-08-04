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



# Expose the gcn_conv_forward function from the C++ extension.
gcn_conv_forward = cpp_ext.gcn_conv_forward


# Now we can define our GNN layers in Python using the C++ function.
# This approach keeps the PyTorch Module structure for automatic parameter
# handling, while offloading the heavy computation to C++.
class GCNConv(torch.nn.Module):
    """
    A Graph Convolutional Network (GCN) layer implemented in Python,
    with the forward pass handled by a custom C++ extension.
    """
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The weight is a trainable parameter, managed by PyTorch.
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj, apply_relu=True):
        # Call the C++ extension function for the forward pass.
        return gcn_conv_forward(x, adj, self.weight, apply_relu)


class GCN(torch.nn.Module):
    """
    A two-layer GCN model implemented in Python, using the custom
    GCNConv layer.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        # Define the two GCNConv layers.
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, x, adj):
        # Pass through the first convolutional layer with ReLU activation.
        x = self.conv1(x, adj, True)
        
        # Pass through the second convolutional layer without ReLU.
        x = self.conv2(x, adj, False)

        # Apply log_softmax to the final output.
        return torch.log_softmax(x, dim=1)


# This `__all__` variable defines the public API of your package.
__all__ = ["gcn_conv_forward", "GCNConv", "GCN"]
