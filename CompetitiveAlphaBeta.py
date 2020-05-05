#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a CompetitiveAlphaBeta class to compute
the lower bound of Player I’s maximum adversary distance
while Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""
from numpy import inf

from FeatureExtraction import *
from GameMoves import *
from basics import *


class CompetitiveAlphaBeta:
    def __init__(self, data_set_name, image, image_index, model, eta, tau, attention=False):
        self.image = image
        self.data_set_name = data_set_name
        self.image_index = image_index
        self.model = model
        self.eta = eta
        self.tau = tau
        self.root_label, self.root_conf = self.model.predict(self.image)
        self.attention = attention

        self.game_moves = GameMoves(model = self.model,
                                    image = self.image,
                                    tau = self.tau,
                                    pixel_bounds = (0, 1),
                                    attention = self.attention)
        self.game_moves.generate_moves(collapse_channels = False)

        self.image_bounds = (0, 1)

        self.alpha = {}
        self.beta = {}
        self.manip_beta = {}
        self.manip_dist = {}
        self.current_manip = ()

        self.robust_feature_found = False
        self.robust_feature = []
        self.fragile_feature_found = False
        self.least_fragile_feature = []

    def target_pixels(self, image, pixels):
        (row, col, chl) = image.shape

        atomic_manipulations = []
        manipulated_images = []
        print(pixels)
        for (x, )y in pixels:
            for z in range(chl):
                atomic = (x, y, z, 1 * self.tau)
                valid, atomic_image = self.apply_atomic_manipulation(image, atomic)
                if valid is True:
                    manipulated_images.append(atomic_image)
                    atomic_manipulations.append(atomic)
                atomic = (x, y, z, -1 * self.tau)
                valid, atomic_image = self.apply_atomic_manipulation(image, atomic)
                if valid is True:
                    manipulated_images.append(atomic_image)
                    atomic_manipulations.append(atomic)
        manipulated_images = np.asarray(manipulated_images)

        probabilities = self.model.model.predict(manipulated_images)
        labels = probabilities.argmax(axis=1)

        for idx in range(len(manipulated_images)):
            if not diffImage(manipulated_images[idx], self.image):
                continue
            dist = self.eta[0](manipulated_images[idx], self.image)
            if labels[idx] != self.label:
                self.manip_beta.update({self.current_manip + atomic_manipulations[idx]: dist})
                self.manip_dist.update({self.current_manip + atomic_manipulations[idx]: dist})
            else:
                self.manip_beta.update({self.current_manip + atomic_manipulations[idx]: inf})
                self.manip_dist.update({self.current_manip + atomic_manipulations[idx]: dist})

    def apply_atomic_manipulation(self, image, atomic):
        atomic_image = image.copy()
        chl = atomic[0:3]
        manipulate = atomic[3]

        if (atomic_image[chl] >= max(self.image_bounds) and manipulate >= 0) or (
                atomic_image[chl] <= min(self.image_bounds) and manipulate <= 0):
            valid = False
            return valid, atomic_image
        else:
            if atomic_image[chl] + manipulate > max(self.image_bounds):
                atomic_image[chl] = max(self.image_bounds)
            elif atomic_image[chl] + manipulate < min(self.image_bounds):
                atomic_image[chl] = min(self.image_bounds)
            else:
                atomic_image[chl] += manipulate
            valid = True
            return valid, atomic_image

    def search(self, image_index):

        for partitionID, pixels in self.game_moves.player_two.items():
            self.manip_beta = {}
            self.manip_dist = {}
            self.current_manip = ()
            print("partition ID:", partitionID)
            self.target_pixels(self.image, pixels)

            while min(self.manip_beta.values()) is inf and min(self.manip_dist.values()) <= self.eta[1]:
                min_dist = min(self.manip_beta.values())
                print("Current min distance:", min_dist)

                print("Adversary not found.")
                mani_distance = copy.deepcopy(self.manip_beta)
                for atom, _ in mani_distance.items():
                    self.manip_beta.pop(atom)
                    self.current_manip = atom
                    self.manip_dist.pop(atom)

                    new_image = copy.deepcopy(self.image)
                    atomic_list = [atom[i:i + 4] for i in range(0, len(atom), 4)]
                    for atomic in atomic_list:
                        valid, new_image = self.apply_atomic_manipulation(new_image, atomic)

                    self.target_pixels(new_image, pixels)

            if min(self.manip_beta.values()) > self.eta[1] or min(self.manip_dist.values()) > self.eta[1]:
                print("Distance:", )
                print("Adversarial distance exceeds distance bound.")
                self.beta.update({partitionID: None})
            elif min(self.manip_beta.values()) is not inf:
                print("Adversary found.")
                adv_mani = min(self.manip_beta, key=self.manip_beta.get)
                print("Manipulations:", adv_mani)
                adv_dist = self.manip_beta[adv_mani]
                print("Distance:", adv_dist)
                self.beta.update({partitionID: [adv_mani, adv_dist]})

        for partitionID, beta in self.manip_beta:
            print(partitionID, beta)
            if beta is None:
                print("Feature %s is robust." % partitionID)
                self.manip_beta.pop(partitionID)
                self.robust_feature_found = True
                self.robust_feature.append(partitionID)
        if self.manip_beta:
            self.fragile_feature_found = True
            self.alpha = max(self.beta, key=self.beta.get)
            self.least_fragile_feature = self.alpha
            print("Among fragile features, the least fragile feature is:\n"
                  % self.least_fragile_feature)