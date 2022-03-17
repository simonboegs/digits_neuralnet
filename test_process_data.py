from process_data import Data
import numpy as np

data = Data()

x_train, y_train = data.get_x_y_train()
print('train')
print(x_train[0])
print(y_train[0])

x_test, y_test = data.get_x_y_test()
print("test")
print(x_test[0])
print(y_test[0])




