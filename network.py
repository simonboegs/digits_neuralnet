import numpy as np
from layer import Layer
from activation_functions import relu, relu_prime, softmax

class Network:
    def __init__(self, inputLen, shape, layers=None):
        self.learning_rate = .2
        self.shape = shape
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

    def calc_errors(self, y):
        errors = [None] * len(self.layers)
        errors[-1] = self.layers[-1].a - y
        for i in range(len(self.layers)-2,-1,-1):
            errors[i] = np.multiply(relu_prime(self.layers[i].z), np.matmul(np.transpose(self.layers[i+1].W), errors[i+1]))
        return errors

    def calc_gradients(self, x, y):
        errors = self.calc_errors(y)
        print("errors",errors)
        b_grads = [None] * len(errors)
        W_grads = [None] * len(errors)
        prev_a = x
        for i in range(len(errors)):
            b_grads[i] = errors[i]
            prev_a_matrix = np.transpose(np.array([prev_a]))
            error_matrix = np.array([errors[i]])
            result = np.matmul(prev_a_matrix, error_matrix)
            W_grads[i] = result.T
            prev_a = self.layers[i].a
        return [W_grads, b_grads]
    
    def calc_mean_gradients(self, x_V, y_V):
        W_grads_total, b_grads_total = self.calc_gradients(x_V[0], y_V[0])
        for i in range(1,len(x_V)):
            W_grads, b_grads = self.calc_gradients(x_V[i], y_V[i])
            for j in range(len(W_grads)):
                W_grads_total[j] += W_grads[j]
                b_grads_total[j] += b_grads[j]
        for j in range(len(W_grads_total)):
            np.divide(W_grads_total[j], len(x_V))
            np.divide(b_grads_total[j], len(x_V))
        return [W_grads_total, b_grads_total]


    def update_weights_biases(self, mean_gradients): 
        pass 


