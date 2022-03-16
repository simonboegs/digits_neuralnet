import numpy as np
import pytest
from functions import relu, relu_prime, softmax, cross_entropy

def test_relu():
    z = np.array([-2, -1, -.5, 0, .5, 1, 2, 10],dtype=float)
    ans = relu(z)
    assert ans[0] == 0
    assert ans[1] == 0
    assert ans[2] == 0 
    assert ans[3] == 0
    assert ans[4] == .5
    assert ans[5] == 1
    assert ans[6] == 2
    assert ans[7] == 10

def test_relu_prime():
    z = np.array([-2, -1, -.5, 0, .5, 1, 2, 10],dtype=float)
    ans = relu_prime(z)
    assert ans[0] == 0
    assert ans[1] == 0
    assert ans[2] == 0
    assert ans[3] == 0
    assert ans[4] == 1
    assert ans[5] == 1
    assert ans[6] == 1
    assert ans[7] == 1

def test_softmax():
    z = np.array([-2,-1,0,1,2], dtype=float)
    ans = softmax(z)
    assert np.sum(ans) == 1
    assert ans[-1] > ans[0]

def test_cross_entropy():
    y = np.array([1, 0, 0])
    a = np.array([.9, .05, .05])
    print(cross_entropy(a, y))

    a = np.array([.45, .45, .1])
    print(cross_entropy(a, y))

    a = np.array([.05, .05, .9])
    print(cross_entropy(a, y))

    a = np.array([.05, .9, .05])
    print(cross_entropy(a, y))

test_cross_entropy()
