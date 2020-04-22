# from __future__ import prilont_function
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')



from keras import backend as K
import sys
import os

from NeuralNetwork import *
from AttentionNetwork import AttentionNetwork
from DataSet import *
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound
from keras.datasets import mnist, cifar10
from FeatureExtraction import FeatureExtraction
import tensorflow as tf
import random
from keras.models import Sequential, load_model, Model
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
# y_test = keras.utils.np_utils.to_categorical(y_test, 10)
# x_test = x_test.astype('float32')
# x_test /= 255

x_test, y_test = DataSet(data_set = "mnist", trainOrTest='test').get_dataset()


# print("Data loaded")

cifar10_attn = AttentionNetwork(data_set = 'mnist')
# cifar10_attn.load_network('models/attention_cifar10_achive_softmax') #n_type="residual")
cifar10_attn.load_network()

model = cifar10_attn
num_partition = 10

fig, axarr = plt.subplots(10, 4, figsize=(10, 20))

for img_index in range(10):
	image = x_test[20+img_index]
	ax = axarr[img_index, 0]

	fe = FeatureExtraction(pattern='attention')
	partitions = fe.get_partitions(image=image,
		model = model,
		num_partition = num_partition)

	fe.plot_saliency_map(image=image,
						partitions=partitions,
						path = "figures/saliencymap_attn_mnist.png",
						ax = ax)


	ax = axarr[img_index, 1]
	fe = FeatureExtraction()
	partitions = fe.get_partitions(image=image,
		model = model,
		num_partition = num_partition)

	fe.plot_saliency_map(image=image, 
						partitions=partitions,
						path = "figures/saliencymap_gb_mnist.png",
						ax = ax)

	ax = axarr[img_index, 2]
	fe = FeatureExtraction(pattern='black-box')
	partitions = fe.get_partitions(image=image,
		model = model,
		num_partition = num_partition)

	fe.plot_saliency_map(image=image, 
						partitions=partitions,
						path = "figures/saliencymap_bb_mnist.png",
						ax = ax)

	ax = axarr[img_index, 3]
	ax.imshow(np.array([[x[0] for x in y] for y in image]))

axarr[0, 0].set_title('Attention')
axarr[0, 1].set_title('Grey-box')
axarr[0, 2].set_title('Black-box')
axarr[0, 3].set_title('Original image')
plt.savefig('figures/saliencymaps_mnist.png')

print("Network trained.")
