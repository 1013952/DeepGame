"""
Interface class from which bound classes inherit.
Defines initialisation, prototypes search() and
defines generic callbacks.

Author: 1013952
"""

from NeuralNetwork import *
from DataSet import *
from basics import *
import numpy as np

class bound:

	def __init__(self, data_set_name, tau, game_type, eta, model = None, attention = False, verbose = 0):
		self.data_set_name = data_set_name
		self.dataSet = DataSet(self.data_set_name, 'testing')
		self.tau = tau

		assert game_type in ['cooperative', 'competitive']
		self.game_type = game_type
		self.eta = eta
		self.verbose = verbose
		self.attention = attention

		# To be overwritten by subclasses
		self.bound_type = 'None'

		if model is None:
			self.model = NeuralNetwork()
		else:
			self.model = model
		self.model.load_network(data_set_name = self.data_set_name)

		if self.verbose == 1:
			print("Dataset is %s." % self.model.data_set_name)
			# self.model.model.summary()

	"""Main search function
	Return format: tuple of
			delta timestamp: total running time
	      float [0, 1]: new confidence in image
	      float [0, 100]: percentage modified
	      float: new L2 distance
	      float: new L1 distance
	      float: new L0 distance
	      int: maximally vulnerable feature (value can be any if cooperative)"""
	def search(self, image_index):
		raise NotImplementedError


	def calc_distances(self, image, image1):

		l2dist = l2Distance().dist(image, image1)
		l1dist = l1Distance().dist(image, image1)
		l0dist = l0Distance().dist(image, image1)
		percent = diffPercent(image, image1)
		if self.verbose == 1:
			print("L2 distance %s" % l2dist)
			print("L1 distance %s" % l1dist)
			print("L0 distance %s" % l0dist)
			print("manipulated percentage distance %s" % percent)

		return l2dist, l1dist, l0dist, percent

    # Callback for when search finds an adversarial example
	def success_callback(self, image_index, adversary):
		image = self.dataSet.get_input(image_index)
		label, conf = self.model.predict(image)
		label_str = self.model.get_label(int(label))

		new_label, new_conf = self.model.predict(adversary)
		new_label_str = self.model.get_label(int(new_label))
    	
		assure_path_exists("%s_pic/%s_%s/" % (self.data_set_name, self.bound_type, self.game_type))
		path0 = "%s_pic/%s_%s/%s_%s_modified_into_%s_with_confidence_%s.png" % (
				self.data_set_name,
				self.bound_type,
				self.game_type,
				image_index,
				label_str,
				new_label_str,
				new_conf)

		self.model.save_input(adversary, path0)
		path0 = "%s_pic/%s_%s/%s_diff.png" % (
				self.data_set_name,
				self.bound_type,
				self.game_type,
				image_index
				)
		self.model.save_input(np.absolute(image - adversary), path0)


		print("\nfound an adversary image within pre-specified bounded computational resource. "
		      "The following is its information: ")
		print("difference between images: %s" % (diffImage(image, adversary)))

		l2dist, l1dist, l0dist, percent = self.calc_distances(image, adversary)

		print("class is changed into '%s' with confidence %s\n" % (new_label_str, new_conf))

		return l2dist, l1dist, l0dist, percent

	# Callback for when no adversarial example is found
	def failure_callback(self):
		print("Failed to find an adversary image within pre-specified bounded computational resource.")
