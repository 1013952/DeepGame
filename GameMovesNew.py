#!/usr/bin/env python

"""
author: Xiaowei Huang
revised by: Min Wu (min.wu@cs.ox.ac.uk)
refactored by: Denitsa Markova (denitsa.markova@stcatz.ox.ac.uk)
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from keras import backend as K
from scipy.stats import truncnorm, norm

from basics import *
from FeatureExtractionNew import *
import collections


############################################################
#
#  initialise possible moves for a two-player game
#
################################################################


class GameMoves:

    def __init__(self, model, image, tau, feature_extraction = None, pixel_bounds=(0, 1), verbose=0):
        self.model = model
        self.image = image
        self.tau = tau
        self.pixel_bounds = pixel_bounds

        if feature_extraction is None:
            self.feature_extraction = GBExtraction()
        else:
            self.feature_extraction = feature_extraction

    def generate_moves(self, collapse_channels = True):
        kps = self.feature_extraction.get_key_points(self.image, num_partition=10)
        partitions = self.feature_extraction.get_partitions(self.image, self.model, num_partition=10)

        img_enlarge_ratio = 1
        image1 = copy.deepcopy(self.image)

        actions = dict()
        actions[0] = kps
        s = 1
        kp2 = []

        if len(image1.shape) == 2:
            image0 = np.zeros(image1.shape)
        else:
            image0 = np.zeros(image1.shape[:2])

        # to compute a partition of the pixels, for an image classification task 
        # partitions = self.getPartition(image1, kps)
        if verbose == 1:
            print("The pixels are partitioned with respect to keypoints.")

        # construct moves according to the obtained the partitions 
        num_of_manipulations = 0
        (_, _, chl) = image1.shape

        for k, blocks in partitions.items():
            all_atomic_manipulations = []

            for block in blocks:
                x = block[0]
                y = block[1]

                if image0[x][y] == 0: 
                    if collapse_channels is True:
                        atomic_pos = {}
                        atomic_neg = {}
                        for ch in range(chl):
                            atomic_pos[(x, y, ch)] = self.tau
                            atomic_neg[(x, y, ch)] = -1 * self.tau
                        all_atomic_manipulations.append(atomic_pos)
                        all_atomic_manipulations.append(atomic_neg)
                    else:
                        for ch in range(chl):
                            all_atomic_manipulations.append({(x, y, ch): self.tau})
                            all_atomic_manipulations.append({(x, y, ch): - self.tau})
                image0[x][y] = 1

            actions[s] = all_atomic_manipulations
            kp2.append(kps[s - 1])

            s += 1

            num_of_manipulations += len(all_atomic_manipulations)

        # index-0 keeps the keypoints, actual actions start from 1
        actions[0] = kp2
        if verbose == 1:
            print("the number of all manipulations initialised: %s\n" % num_of_manipulations)

        self.moves = actions

    def applyManipulation(self, image, manipulation, report_valid=False):
        # apply a specific manipulation to have a manipulated input
        image1 = copy.deepcopy(image)
        lc, hc = self.pixel_bounds
        valid = True

        for (x, y, ch), man in manipulation.items():
            image1[x][y][ch] += man
            if np.max(image1) > hc or np.min(image1) < lc:
                valid = False
                image1 = np.clip(image1, lc, hc)

        if report_valid:
            return valid, image1
        return image1