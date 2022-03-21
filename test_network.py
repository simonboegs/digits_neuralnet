import numpy as np
from network import Network
from layer import Layer
from process_data import Data

filename = 'network_0.json'
network = Network(filename=filename)

data = Data()
x_test, y_test = data.get_x_y_test()

accuracy = network.test(x_test, y_test)
print(accuracy)
