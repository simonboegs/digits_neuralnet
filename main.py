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
network = Network(learning_rate=.2, input_length=inputLen, shape=shape)
x_train_small = x_train[:100]
y_train_small = y_train[:100]
batch_costs = network.train(x_train, y_train, 20, batch_size=100)
print(batch_costs)
network.save('network_0.json')
with open('train_latest.json','w') as f:
    json.dump({'batch_costs': batch_costs}, f)
    print('batch_costs saved')

plt.plot(batch_costs)
plt.show()
