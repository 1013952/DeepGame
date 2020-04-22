"""
Construct a FeatureExtraction class to retrieve
'key points', 'partitions', `saliency map' of an image
in a black-box or grey-box pattern.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk

Refactored: Denitsa Markova
"""

import copy
import numpy as np
import cv2
import random
from scipy.stats import norm
from keras import backend as K
from AttentionNetwork import AttentionNetwork
from itertools import combinations
from matplotlib import pyplot as plt


# Define a Feature Extraction class.
class FeatureExtraction:
    def __init__(self, verbose=0):
        # Verbosity patterns:
        # 0 - no non-error message outputs
        # 1 - logs to stdout
        # 2 - logs to txt file
        self.verbose = verbose

        # black-box parameters
        self.IMG_ENLARGE_RATIO = 1
        self.IMAGE_SIZE_BOUND = 100
        self.MAX_NUM_OF_PIXELS_PER_KEY_POINT = 1000000

        # grey-box parameters
        self.NUM_PARTITION = 10
        self.PIXEL_BOUNDS = (0, 1)
        self.NUM_OF_PIXEL_MANIPULATION = 2


    # Get key points of an image.
    def get_key_points(self, image, num_partition=10):
        return range(self.NUM_PARTITION)


    # Get partitions of an image.
    def get_partitions(self, image, model=None, num_partition=10, pixel_bounds=(0, 1)):
    	raise NotImplementedError

    # Get saliency map of an image.
    def get_saliency_map(self, image, model, pixel_bounds=(0, 1)):
        self.PIXEL_BOUNDS = pixel_bounds

        image_class, _ = model.predict(image)

        new_pixel_list = np.linspace(self.PIXEL_BOUNDS[0], self.PIXEL_BOUNDS[1], self.NUM_OF_PIXEL_MANIPULATION)
        image_batch = np.kron(np.ones((self.NUM_OF_PIXEL_MANIPULATION, 1, 1, 1)), image)

        manipulated_images = []
        (row, col, chl) = image.shape

        for i, j in np.ndindex(row, col):
            # need to be very careful about image.copy()
            changed_image_batch = image_batch.copy()
            for p in range(0, self.NUM_OF_PIXEL_MANIPULATION):
                changed_image_batch[p, i, j, :] = new_pixel_list[p]
            manipulated_images.append(changed_image_batch)  # each loop append [pixel_num, row, col, chl]

        manipulated_images = np.asarray(manipulated_images)  # [row*col, pixel_num, row, col, chl]
        manipulated_images = manipulated_images.reshape(row * col * self.NUM_OF_PIXEL_MANIPULATION, row, col, chl)

        # Use softmax logits instead of probabilities,
        # as probabilities may not reflect precise influence of one single pixel change.
        features_list = model.softmax_logits(manipulated_images)
        feature_change = features_list[:, image_class].reshape(-1, self.NUM_OF_PIXEL_MANIPULATION).transpose()

        min_indices = np.argmin(feature_change, axis=0)
        min_values = np.amin(feature_change, axis=0)
        min_idx_values = min_indices.astype('float32') / (self.NUM_OF_PIXEL_MANIPULATION - 1)

        [x, y] = np.meshgrid(np.arange(row), np.arange(col))
        x = x.flatten('F')  # to flatten in column-major order
        y = y.flatten('F')  # to flatten in column-major order

        target_feature_list = np.hstack((np.split(x, len(x)),
                                         np.split(y, len(y)),
                                         np.split(min_values, len(min_values)),
                                         np.split(min_idx_values, len(min_idx_values))))

        saliency_map = target_feature_list[target_feature_list[:, 2].argsort()]

        return saliency_map

    def plot_saliency_map(self, image, partitions, path, ax = None):
        heatmap = np.zeros(image.shape[0:2])
        for partitionID, pixels in partitions.values():
            for pixel in pixels:
                heatmap[pixel] = partitionID + 1
        if ax is None:
            plt.imsave(path, cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_AREA), cmap='tab10')
        else:
            ax.imshow(cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_AREA), cmap='tab10')




class GBExtraction(FeatureExtraction):

    def get_partitions(self, image, model=None, num_partition=10):
        self.NUM_PARTITION = num_partition
        assert model is not None

        if self.verbose == 1:
            print("Extracting image features using '%s pattern." % self.PATTERN)

        saliency_map = self.get_saliency_map(image, model)

        partitions = {}
        q, r = divmod(len(saliency_map), self.NUM_PARTITION)
        for key in range(self.NUM_PARTITION):
            partitions[key] = [(int(saliency_map[idx, 0]), int(saliency_map[idx, 1])) for idx in
                                   range(key * q, (key + 1) * q)]
            if key == self.NUM_PARTITION - 1:
                partitions[key].extend((int(saliency_map[idx, 0]), int(saliency_map[idx, 1])) for idx in
                                           range((key + 1) * q, len(saliency_map)))
        return partitions



class BBExtraction(FeatureExtraction):

    def get_key_points(self, image, num_partition=10):
    	self.NUM_PARTITION = num_partition

    	image = copy.deepcopy(image)

        sift = cv2.xfeatures2d.SIFT_create()  # cv2.SIFT() # cv2.SURF(400)

        # Rescalee image to [0, 255]
        if np.max(image) <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)


        if max(image.shape) < self.IMAGE_SIZE_BOUND:
            # For a small image, SIFT works by enlarging the image.
            image = cv2.resize(image, (0, 0), fx=self.IMG_ENLARGE_RATIO, fy=self.IMG_ENLARGE_RATIO)
            key_points, _ = sift.detectAndCompute(image, None)
            for i in range(len(key_points)):
                old_pt = (key_points[i].pt[0], key_points[i].pt[1])
                key_points[i].pt = (int(old_pt[0] / self.IMG_ENLARGE_RATIO),
                                        int(old_pt[1] / self.IMG_ENLARGE_RATIO))
        else:
            key_points, _ = sift.detectAndCompute(image, None)

        return key_points 


    def get_partitions(self, image, model=None, num_partition=10):
        self.NUM_PARTITION = num_partition
        self.PIXEL_BOUNDS = pixel_bounds

     	key_points = self.get_key_points(image)

        if self.verbose == 1:
            print("Extracting image features using '%s' pattern." % self.PATTERN)
            print("%s keypoints are found." % (len(key_points)))

        partitions = {}
    	sh = max(image.shape)

        # FOr small images, such as MNIST, CIFAR10
        if sh < self.IMAGE_SIZE_BOUND:
        	for x, y in np.ndindex((sh, sh)):
        		ps = 0
        		maxk = -1
                for i in range(len(key_points)):
                    k = key_points[i - 1]
                    dist2 = np.linalg.norm(np.array([x, y]) - np.array([k.pt[0], k.pt[1]]))
                    ps2 = norm.pdf(dist2, loc=0.0, scale = k.size)
                    if ps2 > ps:
                        ps = ps2
                        maxk = i
                    if maxk in partitions.keys():
                    	partitions[maxk].append((x, y))
                    else:
                        partitions[maxk] = [(x, y)]

			# If a partition gets too many pixels, randomly remove some pixels:
			if self.MAX_NUM_OF_PIXELS_PER_KEY_POINT > 0:
				for mk, val in partitions.items():
					begining_num = len(val)
					for i in range(begining_num - self.MAX_NUM_OF_PIXELS_PER_KEY_POINT):
							val.remove(random.choice(val))

			return partitions
		# For large images, such as ImageNet
		else:
			key_points = key_points[:200]
			each_num = max(image.shape) ** 2 / len(key_points)
			maxk = 1
			martitions[maxk] = []
			for x, y, in np.ndindex((sh, sh)):
				if len(partitions[maxk]) <= each_num:
					partitions[maxk].append((x, y))
				else:
					maxk += 1
					partitions[maxk] = [(x, y)]
			return partitions


class AttExtraction(FeatureExtraction):

    def get_partitions(self, image, model=None, num_partition=10):
        self.NUM_PARTITION = num_partition

        if self.verbose == 1:
            print("Extracting image features using '%s' pattern." % self.PATTERN)

        try:
            map_model = model.get_partition_model()
        except:
            raise Exception("Model must be attention network for attention feature extraction")

        attn = map_model.predict(np.array([image]))[0]
        img_h, img_w, img_ch = image.shape

        # Resize the attention activations to the input image
        attn = cv2.resize(attn, (img_h, img_w))

        if self.NUM_PARTITION > model.n_heads:
        	print("Too few attention heads. Reducing number of partitions to %d" % 
        		model.n_heads)
        	self.NUM_PARTITION = model.n_heads
        else:
        	max_mask = []

        	for _ in range(self.NUM_PARTITION):
        		max_score = -1
        		max_i = -1

        		for i in range(model.n_heads):
        			if not i in max_mask:
        				mask = max_mask + [i]

        				# Score is the sum of the maximal activations within
        				# channels selected by mask
        				score = np.sum(
        					np.max(attn.T[mask], axis=0))

        				if score > max_score:
        					max_score = score
        					max_i = i

        		max_mask.append(max_i)


        	# Extract best-scoring channels and assign partitions
        	imgmap = np.argmax(attn.T[max_mask], axis=0).T
        	partitions = {}

        	for i in max_mask:
        		partitions[i] = [tuple(loc) for loc in np.argwhere(imgmap == i)]

        return partitions
