import numpy as np
from network import Network
from mnist import MNIST
import json

mndata = MNIST('./MNIST')
images_train_list, labels_train_list = mndata.load_training()
images_test_list, labels_test_list = mndata.load_testing()

images_train = np.array(images_train_list)
labels_train = np.array(labels_train_list)
images_test = np.array(images_test_list)
labels_test = np.array(labels_test_list)

def process_input(images):
    inputs = [None] * len(images)
    for i in range(len(images)):
        inputs[i] = images[i] / 255
    return np.array(images)

def process_output(labels):
    outputs = [None] * len(labels)
    for i in range(len(labels)):
        label = int(labels[i])
        output = np.zeros(10, dtype=float)
        output[label] = 1
        outputs[i] = output
    return np.array(outputs)

x_train = process_input(images_train)
y_train = process_output(labels_train)
x_test = process_input(images_test)
y_test = process_input(labels_test)

dict_train = {
        'x': x_train.tolist(),
        'y': y_train.tolist()
        }
json_train = json.dumps(dict_train)
with open('data_train.json', 'w') as outfile:
    json.dump(json_train, outfile)

dict_test = {
        'x': x_test.tolist(),
        'y': y_test.tolist()
        }
json_test = json.dumps(dict_test)
with open('data_test.json', 'w') as outfile:
    json.dump(json_test, outfile)
