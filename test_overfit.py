import numpy as np
from network import Network
from layer import Layer
from mock import mock_network_0 as network

#trying to test overfit here. lets feed in 1 training example and run backprop many times on it and see if our network will converge to the correct output.
x_V_train = [np.array([.1, .5, 1])]
y_V_train = [np.array([1, 0, 0])]

network.learning_rate = .5
for i in range(10):
    print("ITERATION",i)
    network.train_mini_batch(x_V_train, y_V_train)
    network.test(x_V_train, y_V_train)

