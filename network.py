import numpy as np
from layer import Layer
from activation_functions import relu, relu_prime, softmax

class Network:
    def __init__(self, inputLen, shape, layers=None):
        if layers != None:
            self.layers = layers
        else:
            self.layers = []

            for i in range(len(shape)):
                if i == 0:
                    prevN = inputLen
                else:
                    prevN = shape[i-1]
                N = shape[i]

                W = self.init_weight_M(prevN, N)
                b = self.init_bias_V(N)
                a = self.init_activation_V(N)
                z = self.init_z_V(N)

                self.layers.append(Layer(W,b,z,a))

    #INITS
    def init_weight_M(self, prevN, N):
        W = np.random.rand(N, prevN)
        return W

    def init_bias_V(self, N):
        b = np.zeros(N, dtype=float)
        return b

    def init_activation_V(self, N):
        a = np.zeros(N, dtype=float)
        return a

    def init_z_V(self, N):
        z = np.zeros(N, dtype=float)
        return z

    def forward_prop(self, x): #takes input vector
        prev_a = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            #calculate z
            layer.z = np.matmul(layer.W, prev_a) + layer.b
            #calculate a
            if i == len(self.layers) - 1:
                layer.a = softmax(layer.z)
            else:   
                layer.a = relu(layer.z)
            prev_a = layer.a

