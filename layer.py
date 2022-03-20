import numpy as np

class Layer:
    def __init__(self, N, weight_matrix, bias_vector):
        self.N = N
        self.W = weight_matrix
        self.b = bias_vector
        self.a = np.zeros(N)
        self.z = np.zeros(N) 

    def printLayer(self):
        print('LAYER N='+str(self.N))
        print('W', self.W.shape, self.W)
        print('b', self.b)
        print('z', self.z)
        print('a', self.a)

