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
from FeatureExtraction import *
from DataSet import *
import tensorflow as tf
import cv2
import copy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

nn = AttentionNetwork()
extract_method = FeatureExtraction()

fig, axes = plt.subplots(2, 4)
for row in axes:
	for ax in row:
		ax.set_xticks([])
		ax.set_yticks([])

new_extract = extract_method.load_dataset('gb')

for i, data_set_name in enumerate(['cifar10', 'gtsrb']):
	nn.load_network(data_set_name = data_set_name)

	for j, image_index in enumerate([23, 27]):
		image = DataSet(data_set_name = data_set_name, trainOrTest='test').get_input(image_index)
		ax = axes[i, 2*j]
		if image.shape[2] == 1:
			ax.imshow(image.T[0].T, cmap = 'Greys')
		else:
			ax.imshow(image)

		partitions = new_extract.get_partitions(image = image, model = nn, num_partition=10)
		ax = axes[i, 2*j+1]
		print(ax)
		new_extract.plot_saliency_map(image = image, partitions = partitions, path = None, ax = ax)



axes[0, 0].set_title("Original")
axes[0, 1].set_title("Feature \n partitions")
axes[0, 3].set_title("Feature \n partitions")
axes[0, 2].set_title("Original")
plt.savefig("figures/gb_comparison.png")


"""nn.load_network(data_set_name = 'gtsrb')
gtsrb = DataSet(data_set_name = 'gtsrb', trainOrTest = 'test')
new_extract = extract_method.load_dataset('att')



fig, axes = plt.subplots(3, 4)

for row in axes:
	for ax in row:
		ax.set_xticks([])
		ax.set_yticks([])

for i, image_index in enumerate([20, 23, 184]):

	image = gtsrb.get_input(image_index)

	adversary = matplotlib.image.imread('gtsrb_pic/%s_Unsafe_currentBest_1.png' % image_index)
	adversary = adversary[:, :, :3]

	axes[i, 0].imshow(image)

	partitions = new_extract.get_partitions(image = image, model = nn, num_partition = 10)
	new_extract.plot_saliency_map(image = image, partitions = partitions, path = None, ax = axes[i, 1])

	partitions = new_extract.get_partitions(image = adversary, model = nn, num_partition = 10)
	new_extract.plot_saliency_map(image = image, partitions = partitions, path = None, ax = axes[i, 2])
	axes[i, 3].imshow(adversary)

	label, conf = nn.predict(image)
	label_str = nn.get_label(label)

	print("Original: %s, %s" % (label_str, conf))
	new_label, new_conf = nn.predict(adversary)
	new_str = nn.get_label(new_label)
	print("adversary: %s, %s" % (new_str, new_conf))

axes[0, 0].set_title("Original")
axes[0, 1].set_title("Original \n features")
axes[0, 2].set_title("Adversary \n features")
axes[0, 3].set_title("Adversary")
plt.savefig('figures/weirdexample.png')"""

