import os
import numpy as np
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')), name='Cora')
data = dataset[0]


def nll_loss(log_probs, labels):
    """
    log_probs: numpy array of shape (N, C), log-softmax output
    labels: numpy array of shape (N, C), one-hot encoded ground truth
    Returns: scalar loss
    """
    loss = -np.sum(labels * log_probs, axis=1)  # shape: (N,)
    return np.mean(loss)


class ManualGCNLayer():
    def __init__(self, in_dim, out_dim):
        self.X = None
        self.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))  # Xavier init
        
    def forward(self, X, adjm):
        self.X = X  # cache input
        return np.dot(np.dot(adjm, X), self.weights)
    
    def backward(self, Y_grad, adjm, learningRate):
        # dL/dW = (A @ X)^T @ dL/dZ
        A_X = np.dot(adjm, self.X)
        dW = np.dot(A_X.T, Y_grad)

        # dL/dX = A @ (dL/dZ @ W.T)
        dX = np.dot(adjm, np.dot(Y_grad, self.weights.T))

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


class ManualLogSoftmaxLayer():
    def __init__(self):
        self.out = None  # Cache output for backward

    def forward(self, logits):
        # Numerically stable log-softmax
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(logits_stable), axis=1, keepdims=True))
        self.out = logits_stable - logsumexp  # log-softmax output
        return self.out

    def backward(self, y, train_mask=None):
        """
        y: one-hot encoded ground truth labels (shape: [N, C])
        train_mask: boolean array of shape (N,), True for training nodes
        Returns: gradient of loss w.r.t. log-softmax input (shape: [N, C])
        """
        grad = np.exp(self.out) - y  # shape: (N, C)

        if train_mask is not None:
            train_mask = train_mask[:, np.newaxis]  # shape: (N, 1) for broadcasting
            N_train = np.sum(train_mask)
            grad = (grad * train_mask) / N_train  # only average over training samples
        else:
            N = y.shape[0]
            grad = grad / N  # fallback to averaging over all nodes

        return grad


class ManualGCN():
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            learning_rate=0.1,
            dropout_p=0.4
        ):
        # important vars
        self.learning_rate = learning_rate

        # Layers
        self.gcn1 = ManualGCNLayer(input_dim, hidden_dim)
        self.relu = ManualReluLayer()
        self.dropout = ManualDropLayer(p=dropout_p)
        self.gcn2 = ManualGCNLayer(hidden_dim, output_dim)
        self.log_softmax = ManualLogSoftmaxLayer()

    def forward(self, X, adjm, training=True):
        h = self.gcn1.forward(X, adjm)
        h = self.relu.forward(h)
        h = self.dropout.forward(h, training=training)
        h = self.gcn2.forward(h, adjm)
        return self.log_softmax.forward(h)

    def backward(self, y, adjm, train_mask=None):
        grad_logits = self.log_softmax.backward(y, train_mask)
        grad_hidden = self.gcn2.backward(grad_logits, adjm, self.learning_rate)
        grad_hidden = self.dropout.backward(grad_hidden)
        grad_hidden = self.relu.backward(grad_hidden)
        _ = self.gcn1.backward(grad_hidden, adjm, self.learning_rate)

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
    learning_rate=1,
    dropout_p=0
)

adjm = model._normalize_adj(dataset.edge_index, data.num_nodes)

# train one epoch on data, no optimizer now
def train():
    X = data.x.numpy()
    y = np.eye(dataset.num_classes)[data.y.numpy()] # one-hot encoded

    out = model.forward(X=X, adjm=adjm, training=True)
    loss = nll_loss(out[data.train_mask], y[data.train_mask])
    model.backward(y=y, adjm=adjm, train_mask=data.train_mask.numpy())
    
    return loss

def accuracy(preds, labels):
    return np.mean(preds == labels)

def test():
    X = data.x.numpy()
    y = data.y.numpy()
    out = model.forward(X=X, adjm=adjm, training=False)
    pred = np.argmax(out, axis=1)
    return (
        accuracy(pred[data.train_mask], y[data.train_mask]),
        accuracy(pred[data.val_mask], y[data.val_mask]),
        accuracy(pred[data.test_mask], y[data.test_mask]),
    )

# Training loop
for epoch in range(1, 21):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

