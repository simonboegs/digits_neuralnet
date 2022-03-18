import json
import numpy as np
from matplotlib import pyplot as plt
from network import Network
from layer import Layer
from process_data import Data


data = Data()
x_train, y_train = data.get_x_y_train()

inputLen = len(x_train[0])
shape = [16, 16, 10]
network = Network(inputLen, shape)
#for l in range(len(network.layers)):
#    network.layers[l].printLayer()

batch_costs = network.train(x_train, y_train, 1)
print(batch_costs)

plt.plot(batch_costs)
plt.show()
