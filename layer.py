import numpy as np

class Layer:
    def __init__(self, N, weightMatrix, biasVector):
        self.N = N
        self.W = weightMatrix
        self.b = biasVector
        self.a = np.zeros(N)
        self.z = np.zeros(N) 

    def printLayer(self):
        print('LAYER N='+str(N))
        print('W', self.W.shape, self.W)
        print('b', self.b)
        print('z', self.z)
        print('a', self.a)

