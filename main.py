import numpy as np
from network import Network

network = Network(5, [3,3])

for layer in network.layers:
    layer.printLayer()
