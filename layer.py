import numpy as np

class Layer:
    def __init__(self, weightMatrix, biasVector, zVector, activationVector):
        self.W = weightMatrix
        self.b = biasVector
        self.a = activationVector
        self.z = zVector

    def printLayer(self):
        print('LAYER')
        print('W', self.W.shape, self.W)
        print('b', self.b)
        print('z', self.z)
        print('a', self.a)

