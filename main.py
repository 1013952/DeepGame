from __future__ import print_function
from keras import backend as K
import sys
import os

from NeuralNetwork import *
from DataSet import *
from DataCollection import *
from upperboundNew import upperbound
from lowerboundNew import lowerbound
from basicsNew import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# Raise error if number of parameters does not match
assert len(sys.argv) in [1, 9]

# the first way of defining parameters
if len(sys.argv) == 9:

    if sys.argv[1] == 'mnist' or sys.argv[1] == 'cifar10' or sys.argv[1] == 'gtsrb':
        data_set_name = sys.argv[1]
    else:
        print("please specify as the 1st argument the dataset: mnist or cifar10 or gtsrb")
        exit

    if sys.argv[2] == 'ub' or sys.argv[2] == 'lb':
        bound_type = sys.argv[2]
    else:
        print("please specify as the 2nd argument the bound: ub or lb")
        exit

    if sys.argv[3] == 'cooperative' or sys.argv[3] == 'competitive':
        game_type = sys.argv[3]
    else:
        print("please specify as the 3nd argument the game mode: cooperative or competitive")
        exit

    if isinstance(int(sys.argv[4]), int):
        image_index = int(sys.argv[4])
    else:
        print("please specify as the 4th argument the index of the image: [int]")
        exit

    if sys.argv[5] == 'L0':
        distMeasure = l0Distance() 
    elif sys.argv[5] == 'L1':
        distMeasure = l1Distance()
    elif sys.argv[5] == 'L2':
        distMeasure = l2Distance()
    else:
        print("please specify as the 5th argument the distance measure: L0, L1, or L2")
        exit

    if isinstance(float(sys.argv[6]), float):
        dist_cap = float(sys.argv[6])
    else:
        print("please specify as the 6th argument the distance: [int/float]")
        exit
    eta = (distMeasure, dist_cap)
 
    if isinstance(float(sys.argv[7]), float):
        tau = float(sys.argv[7])
    else:
        print("please specify as the 7th argument the tau: [int/float]")
        exit

    if sys.argv[8] == 'y':
        attention = True
    elif sys.argv[8] == 'n':
        attention = False
    else:
        print("please specify as the 8th argument whether to use attention: [y/n]")
        exit

elif len(sys.argv) == 1:
    # the second way of defining parameters
    dataSetName = 'cifar10'
    bound = 'lb'
    gameType = 'cooperative'
    image_index = 213
    eta = ('L2', 10)
    tau = 1
    attention = 0

if bound == 'ub':
    ub_instance = upperbound(data_set = dataSetName,
                                bound = bound,
                                tau = tau,
                                game_type = gameType,
                                eta = eta,
                                attention = attention,
                                model = None,
                                verbose = 1)

    elapsedTime, newConfident, percent, l2dist, l1dist, l0dist, maxFeatures = ub_instance.search(image_index = image_index)

elif bound == 'lb':
    lowerbound(dataSetName, image_index, gameType, eta, tau, attention)

else:
    print("Unrecognised bound setting.\n"
          "Try 'ub' for upper bound or 'lb' for lower bound.\n")
    exit

# dc.provideDetails()
# dc.summarise()
# dc.close()

K.clear_session()
