#!/usr/bin/env python
# Author - Min Wu, minor corrections by 1013952

import numpy as np
import math
import time
import os
import copy


def assure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def current_milli_time():
    return int(round(time.time() * 1000) % 4294967296)


def diffImage(image1, image2):
    return list(zip(*np.nonzero(np.subtract(image1, image2))))


def diffPercent(image1, image2):
    return len(diffImage(image1, image2)) / float(image1.size)


def numDiffs(image1, image2):
    return len(diffImage(image1, image2))


class distMetric:
    def dist(self, image1, image2):
        raise NotImplementedError

    def dtau(self, n):
        raise NotImplementedError

class l0Distance(distMetric):
    def dist(self, image1, image2):
        return np.count_nonzero(np.absolute(np.subtract(image1, image2)))

    def dtau(self, n):
        return 1

class l1Distance(distMetric):
    def dist(self, image1, image2):
        return np.sum(np.absolute(np.subtract(image1, image2)))

    def dtau(self, n):
        return n

class l2Distance(distMetric):
    def dist(self, image1, image2):
        return math.sqrt(np.sum(np.square(np.subtract(image1, image2))))

    def dtau(self, n):
        return math.sqrt(n)

def mergeTwoDicts(x, y):
    epsilon = 0.000000001
    z = x.copy()
    for key, value in y.items():
        if key in z.keys():
            z[key] += y[key]
        else:
            z[key] = y[key]
        if np.abs(z[key]) < epsilon:
            z.pop(key, None)
    # z.update(y)
    return z



def printDict(dictionary):
    for key, value in dictionary.items():
        print("%s : %s" % (key, value))
    print("\n")
