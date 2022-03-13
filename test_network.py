import numpy as np
from network import Network
from layer import Layer

layer0 = Layer(np.array([[0.23464666, 0.55318249, 0.07024911],
 [0.1092703, 0.86004429, 0.10572249],
 [0.99818617, 0.12326535, 0.2342077]]), 
 np.array([0,0,0], dtype=float), 
 np.array([0,0,0], dtype=float),
 np.array([0,0,0], dtype=float))

layer1 = Layer(np.array([[0.6096354, 0.13629155, 0.1139385],
 [0.4286736, 0.3820901, 0.68733547],
 [0.07953902, 0.21710172, 0.19525503]]),
 np.array([0,0,0], dtype=float), 
 np.array([0,0,0], dtype=float), 
 np.array([0,0,0], dtype=float))

shape = [3, 3]
network = Network(3,shape, [layer0,layer1])
network.layers[0].printLayer()
network.layers[1].printLayer()
x = np.array([.1, .5, 1])
print("forward_prop, x =", x)
network.forward_prop(x)
network.layers[0].printLayer()
network.layers[1].printLayer()

y = np.zeros(3)
y[0] = 1

network.train_mini_batch([np.zeros(2),np.zeros(3),np.zeros(4)],[np.ones(2),np.ones(3),np.ones(4)])
