import os
import numpy as np
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')), name='Cora')
data = dataset[0]

# def nll_loss(preds, y_correct):
#     """
#     preds: output of softmax probabilities (shape: [N, C])
#     y_correct: one-hot encoded ground truth labels (shape: [N, C])
#     """
#     eps = 1e-9 # Add epsilon for numerical stability
#     log_preds = np.log(preds + eps)  # shape: [N, C]
#     loss = -np.mean(np.sum(y_correct * log_preds, axis=1))
#     return loss
def nll_loss(preds, labels):
    """
    preds: output of softmax (shape: [N, C])
    labels: ground truth labels (shape: [N])
    """
    N = labels.shape[0]
    log_probs = -np.log(preds[np.arange(N), labels] + 1e-9)  # Add epsilon to avoid log(0)
    return np.mean(log_probs)


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


class ManualSoftmaxLayer():
    def __init__(self):
        self.out = None  # Cache output for backward

    def forward(self, logits):
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.out

    def backward(self, labels):
        N = labels.shape[0]
        grad = self.out.copy()
        grad[np.arange(N), labels] -= 1
        grad /= N
        return grad


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
        self.softmax = ManualSoftmaxLayer()

    def forward(self, X, training=True):
        h = self.gcn1.forward(X)
        h = self.relu.forward(h)
        h = self.dropout.forward(h, training=training)
        h = self.gcn2.forward(h)
        return self.softmax.forward(h)

    def backward(self, loss):
        grad_logits = self.softmax.backward(loss)
        grad_hidden = self.gcn2.backward(grad_logits, self.learning_rate)
        grad_hidden = self.dropout.backward(grad_hidden)
        grad_hidden = self.relu.backward(grad_hidden)
        _ = self.gcn1.backward(grad_hidden, self.learning_rate)

    @staticmethod
    def _normalize_adj(edge_index, num_nodes):
        # Step 1: Initialize adjacency matrix A
        A = np.zeros((num_nodes, num_nodes))
        A[edge_index[0], edge_index[1]] = 1

        # Step 2: Add self-loops: A_hat = A + I
        A += np.eye(num_nodes)

        # Step 3: Degree matrix D
        degree = A.sum(axis=1)
        D_inv_sqrt = np.pow(degree, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

        # Step 4: Normalize: D^-1/2 * A * D^-1/2
        D_mat_inv_sqrt = np.diag(D_inv_sqrt)
        A_norm = D_mat_inv_sqrt @ A @ D_mat_inv_sqrt

        return A_norm


# training + testing
model = ManualGCN(
    dataset.num_node_features,
    16,
    dataset.num_classes,
    dataset.edge_index,
    data.num_nodes
)

# train one epoch on data, no optimizer now
import os
import numpy as np
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')), name='Cora')
data = dataset[0]

# def nll_loss(preds, y_correct):
#     """
#     preds: output of softmax probabilities (shape: [N, C])
#     y_correct: one-hot encoded ground truth labels (shape: [N, C])
#     """
#     eps = 1e-9 # Add epsilon for numerical stability
#     log_preds = np.log(preds + eps)  # shape: [N, C]
#     loss = -np.mean(np.sum(y_correct * log_preds, axis=1))
#     return loss
def nll_loss(preds, labels):
    """
    preds: output of softmax (shape: [N, C])
    labels: ground truth labels (shape: [N])
    """
    N = labels.shape[0]
    log_probs = -np.log(preds[np.arange(N), labels] + 1e-9)  # Add epsilon to avoid log(0)
    return np.mean(log_probs)


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


class ManualSoftmaxLayer():
    def __init__(self):
        self.out = None  # Cache output for backward

    def forward(self, logits):
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.out

    def backward(self, labels):
        # computes grad for both nll loss and softmax together for efficiency
        N = labels.shape[0]
        grad = self.out.copy()
        grad[np.arange(N), labels] -= 1
        grad /= N
        return grad


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
        self.softmax = ManualSoftmaxLayer()

    def forward(self, X, training=True):
        h = self.gcn1.forward(X)
        h = self.relu.forward(h)
        h = self.dropout.forward(h, training=training)
        h = self.gcn2.forward(h)
        return self.softmax.forward(h)

    def backward(self, loss):
        grad_logits = self.softmax.backward(loss)
        grad_hidden = self.gcn2.backward(grad_logits, self.learning_rate)
        grad_hidden = self.dropout.backward(grad_hidden)
        grad_hidden = self.relu.backward(grad_hidden)
        _ = self.gcn1.backward(grad_hidden, self.learning_rate)

    @staticmethod
    def _normalize_adj(edge_index, num_nodes):
        # Step 1: Initialize adjacency matrix A
        A = np.zeros((num_nodes, num_nodes))
        A[edge_index[0], edge_index[1]] = 1

        # Step 2: Add self-loops: A_hat = A + I
        A += np.eye(num_nodes)

        # Step 3: Degree matrix D
        degree = A.sum(axis=1)
        D_inv_sqrt = np.pow(degree, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

        # Step 4: Normalize: D^-1/2 * A * D^-1/2
        D_mat_inv_sqrt = np.diag(D_inv_sqrt)
        A_norm = D_mat_inv_sqrt @ A @ D_mat_inv_sqrt

        return A_norm


# training + testing
model = ManualGCN(
    dataset.num_node_features,
    16,
    dataset.num_classes,
    dataset.edge_index,
    data.num_nodes
)

# train one epoch on data, no optimizer now
def train():
    X = data.x.numpy()
    y = data.y.numpy()

    out = model.forward(X, training=True)
    loss = nll_loss(out[data.train_mask], y[data.train_mask])
    model.backward(y[data.train_mask])

    return loss

# Training loop
for epoch in range(1, 21):
    loss = train()
    # train_acc, val_acc, test_acc = test()
    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
        # print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")