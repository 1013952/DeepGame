from __future__ import print_function
from keras import backend as K
import sys
import os

from NeuralNetwork import *
from AttentionNetwork import *
from DataSet import *
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cifar10_network = AttentionNetwork()
cifar10_network.train_network()

print("Network trained.")