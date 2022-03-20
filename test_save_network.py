import numpy as np
from network import Network
from layer import Layer
from mock import mock_network_0 as network

network.save('mock_network_save.json')

network_new = Network(filename='mock_network_save.json')
network_new.printNetwork()
