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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)
x_test = x_test.astype('float32')
x_test /= 255


print("Data loaded")

cifar10_attn = AttentionNetwork(data_set = 'cifar10')
cifar10_attn.train_network(n_type="residual") #n_type="residual")

# model = cifar10_attn.model
# mask_model = cifar10_attn.mask_model

# # mask_model.summary()

# attn_activations = mask_model.get_layer('activation_29').output
# mask_model = Model(inputs = mask_model.input, outputs=attn_activations)

# batch_size = 128
# num_classes = 10
# epochs = 50
# img_rows, img_cols, img_chls = 32, 32, 3

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
# input_shape = (img_rows, img_cols, img_chls)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

# to_show = mask_model.predict(x_test[20:30])
# to_show = [cv2.resize(im, (32, 32)) for im in to_show]
# to_show = [im/np.sum(im) for im in to_show]
# fig = plt.figure(figsize=(30,14))
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1)
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.imshow(to_show[0].T[i].reshape(32, 32).T, cmap="binary_r")

# plt.savefig("test.png")

# fig = plt.figure(figsize=(30, 7))
# from itertools import combinations 
# def rSubset(arr, r): 
  
#     # return list of all subsets of length r 
#     # to deal with duplicate subsets use  
#     # set(list(combinations(arr, r))) 
#     return list(combinations(arr, r)) 

# max_index = 0
# max_score = 0
# fig = plt.figure(figsize=(7, 30))

# for j in range(10):
# 	print("Separating image ", j)
# 	ax = fig.add_subplot(10, 3, 3*j + 1)
# 	max_mask = [True for i in range(64)]
# 	max_score = -1

# 	im = to_show[j].T #/np.sum(to_show[j].T, axis=0)
# 	max_mask = np.argsort(np.argsort(im, #to_show[j].T,
# 	 	axis=0).reshape(64, 32*32).T.sum(axis=0))[-5:]
# 	print(max_mask)
# 	# for index in range(2000000):
# 	# 	i = random.sample(range(64), 5)
# 	# 	mask = [a in i for a in range(64)]
# 	# 	score = np.sum(np.max(to_show[j].T[mask], axis=0).T)
# 	# 	if score > max_score:
# 	# 		max_mask = mask
# 	# 		max_score = score

# 	ax.imshow(np.argmax(to_show[j].T[max_mask], axis=0).T, cmap='Dark2')
# 	ax = fig.add_subplot(10, 3, 3*j + 2)
# 	ax.imshow(np.sum(to_show[j].T, axis=0).T, cmap='plasma')

# 	ax = fig.add_subplot(10, 3, 3*j + 3)
    
# 	ax.imshow(x_test[20+j])


# plt.savefig("feature_sep.png")




# score = cifar10_attn.model.evaluate(x_test, y_test, verbose=0)
# print("Test loss for attn:", score[0])
# print("Test accuracy for attn:", score[1])
# feature_extraction_attn = FeatureExtraction(pattern = "attention")



# cifar_10 = NeuralNetwork(data_set = 'cifar10')
# cifar_10.load_network()
# score = cifar_10.model.evaluate(x_test, y_test, verbose=0)
# print("Test loss for vanilla:", score[0])
# print("Test accuracy for vanilla:", score[1])

# feature_extraction = FeatureExtraction(pattern = 'grey-box')

# for img in x_train[2:3]:
# 	a = feature_extraction_attn.get_partitions(img, model=cifar10_attn)
# 	b = feature_extraction.get_partitions(img, model=cifar_10)

# 	print(a)
# 	print(b)

# cifar10_attn = AttentionNetwork(data_set = 'cifar10')
# cifar10_attn.train_network()
# print("Saving network")
# cifar10_attn.save_network()

print("Network trained.")
