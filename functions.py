import numpy as np

def relu(z):
    ans = np.array([0] * len(z), dtype=float)
    for i in range(len(z)):
        ans[i] = max(0.0, z[i])
    return ans

def relu_prime(z):
    ans = np.array([0] * len(z), dtype=float)
    for i in range(len(z)):
        if z[i] > 0:
            ans[i] = 1
        else:
            ans[i] = 0
    return ans

def softmax(z):
    e = np.exp(z)
    return e / e.sum()

def cross_entropy(a, y):
    return -1 * np.dot(np.log(a), y)
