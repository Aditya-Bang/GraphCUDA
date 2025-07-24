import os
import numpy as np
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')), name='Cora')
data = dataset[0]


class ManualGCNLayer():
    def __init__(self, in_dim, out_dim, adjm):
        self.X = None
        self.adjm = adjm  # cache for backward
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))  # Xavier init
        
    def forward(self, X):
        self.X = X  # cache input
        return np.dot(np.dot(self.adjm, X), self.weights)
    
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
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            edge_index,
            num_nodes,
            learning_rate=0.01,
            dropout_p=0.5
        ):
        # important vars
        self.adjm = self._normalize_adj(edge_index, num_nodes)
        self.learning_rate = learning_rate

        # Layers
        self.gcn1 = ManualGCNLayer(input_dim, hidden_dim, self.adjm)
        self.relu = ManualReluLayer()
        self.dropout = ManualDropLayer(p=dropout_p)
        self.gcn2 = ManualGCNLayer(hidden_dim, output_dim, self.adjm)

        # Cache for backward
        self.x_input = None
        self.hidden = None
        self.logits = None

    def forward(self, X, training=True):
        self.x_input = X
        h = self.gcn1.forward(X)
        h = self.relu.forward(h)
        h = self.dropout.forward(h, training=training)
        self.hidden = h  # Cache for backprop

        logits = self.gcn2.forward(h)
        self.logits = logits
        return self._softmax(logits)

    #     x = data.x
    #     x = self.gcn1(x, self.adjm)
    #     x = F.relu(x)
    #     x = F.dropout(x, training=self.training)
    #     x = self.gcn2(x, self.adjm)
    #     return F.log_softmax(x, dim=1)
    

    def backward(self, labels):
        # Forward softmax (already cached)
        probs = self._softmax(self.logits)
        N = labels.shape[0]

        # Compute gradient of loss (cross entropy): (softmax - one-hot)
        grad_logits = probs
        grad_logits[np.arange(N), labels] -= 1
        grad_logits /= N

        # Backprop GCN2
        grad_hidden = self.gcn2.backward(grad_logits, self.learning_rate)

        # Backprop Dropout
        grad_hidden = self.dropout.backward(grad_hidden)

        # Backprop ReLU
        grad_hidden = self.relu.backward(grad_hidden)

        # Backprop GCN1
        _ = self.gcn1.backward(grad_hidden, self.learning_rate)

    @staticmethod
    def _softmax(logits):
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

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
