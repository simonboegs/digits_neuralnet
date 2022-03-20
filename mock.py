import numpy as np  
from network import Network
from layer import Layer



mock_W_0 = np.array([[0.23464666, 0.55318249, 0.07024911],
 [0.1092703, 0.86004429, 0.10572249],
 [0.99818617, 0.12326535, 0.2342077]])

mock_W_1 = np.array([[0.6096354, 0.13629155, 0.1139385],
 [0.4286736, 0.3820901, 0.68733547],
 [0.07953902, 0.21710172, 0.19525503]])

mock_b = np.array([0,0,0], dtype=float)


layer0 = Layer(3, mock_W_0, mock_b)
layer1 = Layer(3, mock_W_1, mock_b)
mock_network_0 = Network(layer0, layer1, learning_rate=.1, input_length=3)
