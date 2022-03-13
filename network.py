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

    def update_weights_biases(self, W_grads, b_grads):
        for l in range(len(self.layers)):
            W_grad_adj = np.multiply(self.learning_rate, W_grads[l])
            b_grad_adj = np.multiply(self.learning_rate, b_grads[l])
            self.layers[i].W = np.subtract(self.layers[i].W, W_grad_adj)
            self.layers[i].b = np.subtract(self.layers[i].b, b_grad_adj)
            #z and a are never being reset - not technically a problem, but might be confusing when printing

    def train_mini_batch(self, x_V, y_V):
        W_grads_total = None
        b_grads_total = None
        for i in range(len(x_V)):
            x = x_V[i]
            y = y_V[i]
            self.forward_prop(x) 
            W_grads, b_grads = self.calc_gradients(x, y)
            #init total gradients
            if W_grads_total == None:
                W_grads_total = W_grads.copy()
                b_grads_total = b_grads.copy()
            #add to total gradients
            else:
                for l in range(len(self.layers)):
                    W_grads_total[l] = np.add(W_grads_total[l], W_grads[l])
                    b_grads_total[l] = np.add(b_grads_total[l], b_grads[l])
        #calculate mean grads
        W_grads_mean = [None] * len(self.layers)
        b_grads_mean = [None] * len(self.layers)
        for l in range(len(self.layers)):
            W_grads_mean[l] = np.divide(W_grads_total[l], len(x_V))
            b_grads_mean[l] = np.divide(b_grads_total[l], len(x_V))
        
        self.update_weights_biases(W_grads_mean, b_grads_mean)

