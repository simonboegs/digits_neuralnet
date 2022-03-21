import json
import numpy as np
from layer import Layer
from functions import relu, relu_prime, softmax, cross_entropy

class Network:
    def __init__(self, *layers, **kwargs):
        if len(layers) != 0:
            self.learning_rate = kwargs['learning_rate']
            self.input_length = kwargs['input_length']
            self.layers = layers
            return
        if 'filename' in kwargs:
            with open(kwargs['filename'], 'r') as f:
                network_data = json.load(f)
                self.learning_rate = network_data['learning_rate']
                self.input_length = network_data['input_length']
                self.layers = [None] * len(network_data['layers'])
                for l, layer in enumerate(network_data['layers']):
                    N = layer['N']
                    W = np.asarray(layer['W'])
                    b = np.asarray(layer['b'])
                    self.layers[l] = Layer(N, W, b)
            return
        self.learning_rate = kwargs['learning_rate']
        self.layers = [None] * len(kwargs['shape'])
        self.input_length = kwargs['input_length']
        prevN = self.input_length 
        for l in range(len(kwargs['shape'])):
            N = kwargs['shape'][l] 
            
            W = self.init_weight_M(prevN, N)
            b = self.init_bias_V(N)

            self.layers[l] = Layer(N, W, b)
            prevN = N

    #INITS
    def init_weight_M(self, prevN, N):
        #W = np.random.rand(N, prevN)
        W = np.random.normal(0, 1, (N, prevN))
        return W

    def init_bias_V(self, N):
        b = np.zeros(N, dtype=float)
        return b

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
            for j in range(len(x_batches)):
                print("EPOCH",i,"BATCH",j)
                batch_mean_cost = self.train_batch(x_batches[j], y_batches[j])
                batch_mean_costs.append(batch_mean_cost) 
        return batch_mean_costs
    
    def save(self, filename):
        layers = [None] * len(self.layers)
        for l in range(len(self.layers)):
            layers[l] = {
                    'N': self.layers[l].N,
                    'W': self.layers[l].W.tolist(),
                    'b': self.layers[l].b.tolist()
                    }
        json_obj = {'input_length': self.input_length, 'learning_rate': self.learning_rate, 'layers': layers}
        with open(filename,'w') as f:
            json.dump(json_obj, f)
        print('network saved to file',filename)

    def test(self, x_V, y_V):
        if not self.valid_inputs(x_V, y_V):
            return
        total_correct = 0.0
        for i in range(len(x_V)):
            self.forward_prop(x_V[i])
            a = self.layers[-1].a
            choice = a.argmax()
            correct_choice = y_V[i].argmax()
            if choice == correct_choice:
                total_correct += 1
        return total_correct / len(x_V)
            
    def printNetwork(self):
        print('NETWORK')
        print('learning_rate',self.learning_rate)
        print('input_length',self.input_length)
        for l in range(len(self.layers)):
            self.layers[l].printLayer()
