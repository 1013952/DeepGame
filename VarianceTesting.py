	# from __future__ import prilont_function
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')



from keras import backend as K
import sys
import os
import seaborn as sns
import scipy.stats

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
import copy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

class Node:

	def __init__(self, state, depth, confidence):
		self.state = state
		self.depth = depth
		self.confidence = confidence



class VarianceTesting:

	def __init__(self, network, image, image_ch, feature_extraction, num_partition=10):
		self.network = network
		self.image = image
		self.feature_extraction = feature_extraction
		self.num_partition = num_partition
		self.image_ch = image_ch

		self.keypoints = {}

	def apply_atomic(self, image, loc, atomic_index, tau=1):
		img_copy = copy.deepcopy(image)
		man_directions = np.binary_repr(atomic_index, width=self.image_ch)
		man_directions = np.array([-1 if a==0 else 1 for a in man_directions])

		x, y = loc

		img_copy[x][y] = img_copy[x][y] + man_directions
		img_copy = np.clip(img_copy, 0, 1)
		return img_copy 



	def tree_expand(self, depth = 2):
		c, conf = self.network.predict(self.image)
		print("Initial image:")
		print(c, conf)
		self.root_node = Node(self.image, 0, conf)

		partitions = self.feature_extraction.get_partitions(image=self.image,
													model = self.network,
													num_partition= self.num_partition)

		self.lvl = []
		for k, locs in partitions.items():
			lvl_two = []
			for atomic_index in range(np.power(2, self.image_ch)):
				for loc in locs:
					manipulated = self.apply_atomic(self.image, loc, atomic_index)
					c, conf = self.network.predict(manipulated)
					# print(c, conf)
					lvl_two.append(conf)
			mean = np.mean(lvl_two)
			var = np.var(lvl_two)
			self.lvl.append((mean, var))
			# print("Overall for component:")
			# print(mean, var)

		return np.mean([var for (mean, var) in self.lvl])




# Create the network we will be testing
nn = AttentionNetwork(data_set = 'cifar10')
nn.load_network()

# Load the test dataset
# Initially we will only test on one image
x_test, y_test = DataSet(data_set = "cifar10", trainOrTest='test').get_dataset()


scores = nn.model.evaluate(x_test, y_test, verbose=1)


att_moves = FeatureExtraction(pattern='attention')
gb_moves = FeatureExtraction(pattern='black-box')

atts = []
gbs = []
for index, image in enumerate(x_test[:200]):
	print("Testing image %d" % index)
	# Test first-level variance on attention and grey-box
	vt = VarianceTesting(nn, image, 3, att_moves)
	atts.append(vt.tree_expand())

	vt = VarianceTesting(nn, image, 3, gb_moves)
	gbs.append(vt.tree_expand())



p = sns.regplot(gbs, atts)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()[0].get_ydata())
plt.title("Node variance regression: slope %f and intercept %f" % (slope, intercept))
plt.ylabel("Attention feature extraction")
plt.xlabel("Grey-box feature extraction")

plt.savefig("figures/var_testing_bb.png")



