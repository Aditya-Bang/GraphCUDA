import os
import numpy as np
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')), name='Cora')
data = dataset[0]


class ManualGCNLayer():
    def __init__(self, in_dim, out_dim):
        self.X = None
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))  # Xavier init
        self.adjm = None  # cache for backward

    def forward(self, X, adjm):
        self.X = X  # cache input
        self.adjm = adjm  # cache adjacency
        return np.dot(np.dot(adjm, X), self.weights)
    
    def backward(self, Y_grad, learningRate):
        # dL/dW = (A @ X)^T @ dL/dZ
        A_X = np.dot(self.adjm, self.X)
        dW = np.dot(A_X.T, Y_grad)

        # dL/dX = A @ (dL/dZ @ W.T)
        dX = np.dot(self.adjm, np.dot(Y_grad, self.weights.T))

        # Gradient descent step
        self.weights -= learningRate * dW

        return dX  # Return gradient to propagate to earlier layers

class ManualReluLayer():
    def __init__(self):
        self.mask = None  # To store which values were positive

    def forward(self, X):
        self.mask = (X > 0).astype(float)  # 1 where X > 0, else 0
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

class ManualDropLayer():
    def __init__(self, p=0.5):
        self.p = p  # Dropout probability
        self.mask = None

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(float) / (1.0 - self.p)
            return X * self.mask
        else:
            return X  # No dropout during inference

    def backward(self, grad_output):
        return grad_output * self.mask

class ManualGCN():
    def __init__(self, input_dim, hidden_dim, output_dim, edge_index, num_nodes):
        pass
    #     super(ManualGCN, self).__init__()

    #     self.adjm = self._normalize_adj(edge_index, num_nodes)

    #     self.gcn1 = ManualGCNLayer(input_dim, hidden_dim)
    #     self.gcn2 = ManualGCNLayer(hidden_dim, output_dim)

    def forward(self, data):
        pass
    #     x = data.x
    #     x = self.gcn1(x, self.adjm)
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)
    #     x = self.gcn2(x, self.adjm)
    #     return F.log_softmax(x, dim=1)
    def backward(self):
        pass

    @staticmethod
    def _normalize_adj(edge_index, num_nodes):
        # Step 1: Initialize adjacency matrix A
        A = np.zeros((num_nodes, num_nodes))
        A[edge_index[0], edge_index[1]] = 1

        # Step 2: Add self-loops: A_hat = A + I
        A += np.eye(num_nodes)

        # Step 3: Degree matrix D
        degree = A.sum(dim=1)
        D_inv_sqrt = np.pow(degree, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

        # Step 4: Normalize: D^-1/2 * A * D^-1/2
        D_mat_inv_sqrt = np.diag(D_inv_sqrt)
        A_norm = D_mat_inv_sqrt @ A @ D_mat_inv_sqrt

        return A_norm
