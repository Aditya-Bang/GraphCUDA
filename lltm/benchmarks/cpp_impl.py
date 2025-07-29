import time
import torch
from python.lltm import LLTM

def run_test():
    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)

    rnn = LLTM(input_features, state_size)

    forward = 0
    backward = 0
    for i in range(10000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        if (i > 10): # warmup
            forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        if (i > 10):
            backward += time.time() - start

    print('Forward: {:.4f} s | Backward {:.4f} s'.format(forward, backward))

if __name__ == "__main__":
    run_test()
