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
from FeatureExtraction import *
import collections


############################################################
#
#  initialise possible moves for a two-player game
#
################################################################


class GameMoves:

    def __init__(self, model, image, tau, feature_extraction = None, pixel_bounds=(0, 1), verbose=0, attention = True):
        self.model = model
        self.image = image
        self.tau = tau
        self.pixel_bounds = pixel_bounds
        self.verbose = verbose

        if feature_extraction is None:
            if attention:
                self.feature_extraction = AttExtraction()
            else:
                self.feature_extraction = GBExtraction()
        else:
            self.feature_extraction = feature_extraction


    def generate_moves(self, collapse_channels = True):
        kps = self.feature_extraction.get_key_points(self.image, num_partition=10)
        partitions = self.feature_extraction.get_partitions(self.image, self.model, num_partition=10)

        img_enlarge_ratio = 1
        image1 = copy.deepcopy(self.image)

        actions = dict()
        self.player_one = kps
        kp2 = []

        visited = set()

        # construct moves according to the obtained the partitions 
        num_of_manipulations = 0
        (_, _, chl) = image1.shape

        for k, blocks in partitions.items():
            atomics = []

            for block in blocks:
                x = block[0]
                y = block[1]

                if (x, y) not in visited: 
                    if collapse_channels is True:
                        atomic_pos = {}
                        atomic_neg = {}
                        for ch in range(chl):
                            atomic_pos[(x, y, ch)] = self.tau
                            atomic_neg[(x, y, ch)] = -1 * self.tau
                        atomics.append(atomic_pos)
                        atomics.append(atomic_neg)
                    else:
                        for ch in range(chl):
                            atomics.append({(x, y, ch): self.tau})
                            atomics.append({(x, y, ch): - self.tau})

                visited.add((x, y))

            actions[k] = atomics
            num_of_manipulations += len(atomics)

        # index-0 keeps the keypoints, actual actions start from 1
        self.player_two = actions
        if self.verbose == 1:
            print("the number of all manipulations initialised: %s\n" % num_of_manipulations)

        self.moves = {1: self.player_one, 2: self.player_two}

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