import json
import numpy as np
from layer import Layer
from functions import relu, relu_prime, softmax, cross_entropy

class Network:
    def __init__(self, inputLen, shape, layers=None):
        self.learning_rate = .1
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

                self.layers.append(Layer(N,W,b))

"""
    def __init__(self, input_length, learning_rate=.1, *layers, **kwargs):
        self.learning_rate = learning_rate
        self.input_length = input_length
        if len(layers) != 0:
            self.layers = layers
            self.shape = [len(layer.a) for layer in self.layers]
        elif 'filename' in kwargs:
            with open(filename, 'r') as f:
                data = json.load(f)
                print(data)
        else:
            self.shape = kwargs.shape
            self.layers = [None] * len(shape) 
            prevN = self.input_length 
            for l in range(len(shape)):
                N = self.shape[l] 
                
                W = self.init_weight_M(prevN, N)
                b = self.init_bias_V(N)
                a = self.init_activation_V(N)
                z = self.init_z_V(N)

                prevN = N
                self.layers[l] = Layer(W,b,z,a)
"""
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
            self.layers[i].printLayer()

    def calc_errors(self, y):
        errors = [None] * len(self.layers)
        errors[-1] = self.layers[-1].a - y
        for i in range(len(self.layers)-2,-1,-1):
            errors[i] = np.multiply(relu_prime(self.layers[i].z), np.matmul(np.transpose(self.layers[i+1].W), errors[i+1]))
        return errors

    def calc_gradients(self, x, y):
        errors = self.calc_errors(y)
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
            self.layers[l].W = np.subtract(self.layers[l].W, W_grad_adj)
            self.layers[l].b = np.subtract(self.layers[l].b, b_grad_adj)
            #z and a are never being reset - not technically a problem, but might be confusing when printing

    def valid_inputs(self, x_V, y_V):
        if len(x_V) == len(y_V):
            return True
        print("x_V and y_V have unequal lengths")
        return False

    def train_batch(self, x_V, y_V):
        total_cost = 0
        if not self.valid_inputs(x_V, y_V):
            return
        W_grads_total = None
        b_grads_total = None
        for i in range(len(x_V)):
            x = x_V[i]
            y = y_V[i]
            self.forward_prop(x) 
            cost = cross_entropy(self.layers[-1].a, y)
            total_cost += cost
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
        
        mean_cost = total_cost / len(x_V)
        return mean_cost

    def train(self, x_V_all, y_V_all, epochs, batch_size=500):
        if not self.valid_inputs(x_V_all, y_V_all):
            return
        #split into mini-batches
        batch_num = np.ceil(len(x_V_all)/batch_size)
        x_batches = np.array_split(x_V_all, batch_num)
        y_batches = np.array_split(y_V_all, batch_num)
        batch_mean_costs = []
        for i in range(epochs):
            print("EPOCH", i)
            for j in range(len(x_batches)):
                print("BATCH",j)
                batch_mean_cost = self.train_batch(x_batches[j], y_batches[j])
                batch_mean_costs.append(batch_mean_cost) 
        return batch_mean_costs
    
    def save(self, filename):
        layers = [None] * len(self.layers)
        for l in range(len(self.layers)):
            layers[l] = {
                    'W': self.layers[l].W.tolist(),
                    'b': self.layers[l].b.tolist()
                    }
        json_obj = {input_length: self.input_length, shape: self.shape, layers: layers}
        with open(filename,'w') as f:
            json.dump(json_obj, f)
        print('network saved to file',filename)

    def test(self, x_V, y_V):
        if not self.valid_inputs(x_V, y_V):
            return
        for i in range(len(x_V)):
            self.forward_prop(x_V[i])
            a = self.layers[-1].a
            cost = cross_entropy(a, y_V[i])

    def printNetwork(self):
        print('NETWORK')
        print('learning rate',learning_rate)
        print('shape',shape)
        for l in range(len(self.layers)):
            self.layers[l].printLayer()
