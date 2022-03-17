import numpy as np
from mnist import MNIST
import json

class Data:
    def __init__(self):
        self.mndata = MNIST('./MNIST')

    def process_input(self, images):
        inputs = [None] * len(images)
        for i in range(len(images)):
            inputs[i] = np.divide(images[i], 255)
        return np.array(inputs)

    def process_output(self, labels):
        outputs = [None] * len(labels)
        for i in range(len(labels)):
            label = int(labels[i])
            output = np.zeros(10, dtype=float)
            output[label] = 1
            outputs[i] = output
        return np.array(outputs)

    def get_x_y_train(self):
        images_train_list, labels_train_list = self.mndata.load_training()
        images_train = np.array(images_train_list)
        labels_train = np.array(labels_train_list)
        x_train = self.process_input(images_train)
        y_train = self.process_output(labels_train)
        return (x_train, y_train)

    def get_x_y_test(self):
        images_test_list, labels_test_list = self.mndata.load_testing()
        images_test = np.array(images_test_list)
        labels_test = np.array(labels_test_list)
        x_test = self.process_input(images_test)
        y_test = self.process_output(labels_test)
        return (x_test, y_test)
