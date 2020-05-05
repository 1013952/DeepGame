# !/usr/bin/env python

"""
A data structure for organising search

author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy
import sys
import operator
import random
import math

from basics import *
from GameMoves import *

MCTS_multi_samples = 3
effectiveConfidenceWhenChanging = 0.0
explorationRate = math.sqrt(2)


class MCTSCooperative:

    def __init__(self, data_set, model, image_index, image, tau, eta, attention=False):
        self.data_set = data_set
        self.image_index = image_index
        self.image = image
        self.model = model
        self.tau = tau
        self.eta = eta
        self.attention = attention

        (self.originalClass, self.originalConfident) = self.model.predict(self.image)

        self.moves = GameMoves(model = self.model, 
                                image = self.image, 
                                tau = self.tau,
                                pixel_bounds = (0, 1),
                                attention = self.attention,
                                verbose = 1)
        # Ensure that the GameMoves object has actions initialised
        self.moves.generate_moves(collapse_channels = True)
        if self.verbose == 1:
            print("%s actions have been initialised." % (len(self.actions)))


        self.cost = {}
        self.numberOfVisited = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}

        self.indexToNow = 0
        # current root node
        self.rootIndex = 0

        self.manipulation = {}
        # initialise root node
        self.manipulation[-1] = {}
        self.initialiseLeafNode(0, None, {})

        # record all the keypoints: index -> kp
        self.keypoints = {}
        # mapping nodes to keypoints
        self.keypoint = {}
        self.keypoint[0] = 0

        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (2 ^ 20, {})
        self.numConverge = 0

        # number of adversarial examples
        self.numAdv = 0

        # how many sampling is conducted
        self.numOfSampling = 0

        # temporary variables for sampling 
        self.atomicManipulationPath = []
        self.depth = 0
        self.availableActionIDs = []
        self.usedActionIDs = []


    def initialiseLeafNode(self, index, parentObject, newAtomicManipulation):
        print("initialising a leaf node %s from the node %s" % (index, parentIndex))
        new_leaf = Node(index = index,
                        manipulation = mergeTwoDicts(self.manipulation[parentIndex], newAtomicManipulation),
                        )

    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self, newRootObject):
        if newRootObject.player_to_act == 1:
            player = "the first player"
        else:
            player = "the second player"
        if self.verbose == 1:
            print("%s making a move into the new root %s, whose value is %s and visited number is %s" % (
                player, newRootIndex, self.cost[newRootIndex], self.numberOfVisited[newRootIndex]))
        
        self.removeChildren(self.rootIndex, [newRootIndex])
        self.rootIndex = newRootIndex

    def removeChildren(self, index, indicesToAvoid):
        if self.fullyExpanded[index] is True:
            for childIndex in self.children[index]:
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex, [])
        self.manipulation.pop(index, None)
        self.cost.pop(index, None)
        self.parent.pop(index, None)
        self.keypoint.pop(index, None)
        self.children.pop(index, None)
        self.fullyExpanded.pop(index, None)
        self.numberOfVisited.pop(index, None)

    def bestChild(self, index):
        allValues = {}
        for childIndex in self.children[index]:
            allValues[childIndex] = float(self.numberOfVisited[childIndex]) / self.cost[childIndex]
        print("finding best children from %s" % allValues)
        # for cooperative
        return max(allValues.items(), key=operator.itemgetter(1))[0]

    def treeTraversal(self, index):
        if self.fullyExpanded[index] is True:
            print("tree traversal on node %s with children %s" % (index, self.children[index]))
            allValues = {}
            for childIndex in self.children[index]:
                # UCB values
                allValues[childIndex] = ((float(self.numberOfVisited[childIndex]) / self.cost[childIndex]) * self.eta[1]
                                         + explorationRate * math.sqrt(
                            math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex])))

            # for cooperative
            nextIndex = np.random.choice(list(allValues.keys()), 1,
                                         p=[x / sum(allValues.values()) for x in allValues.values()])[0]

            if self.keypoint[index] in self.usedActionsID.keys() and self.keypoint[index] != 0:
                self.usedActionsID[self.keypoint[index]].append(self.indexToActionID[index])
            elif self.keypoint[index] != 0:
                self.usedActionsID[self.keypoint[index]] = [self.indexToActionID[index]]

            return self.treeTraversal(nextIndex)

        else:
            print("tree traversal terminated on node %s" % index)
            availableActions = copy.deepcopy(self.actions)
            # for k in self.usedActionsID.keys():
            #    for i in self.usedActionsID[k]: 
            #        availableActions[k].pop(i, None)
            return index, availableActions

    def usefulAction(self, ampath, am):
        newAtomicManipulation = mergeTwoDicts(ampath, am)
        activations0 = self.moves.applyManipulation(self.image, ampath)
        (newClass0, newConfident0) = self.model.predict(activations0)
        activations1 = self.moves.applyManipulation(self.image, newAtomicManipulation)
        (newClass1, newConfident1) = self.model.predict(activations1)
        if abs(newConfident0 - newConfident1) < 10 ^ -6:
            return False
        else:
            return True

    def initialiseExplorationNode(self, index, availableActions):
        print("expanding %s" % index)
        if self.keypoint[index] != 0:
            for (actionId, am) in availableActions[self.keypoint[index]].items():
                if self.usefulAction(self.manipulation[index], am) == True:
                    self.indexToNow += 1
                    self.keypoint[self.indexToNow] = 0
                    self.indexToActionID[self.indexToNow] = actionId
                    self.initialiseLeafNode(self.indexToNow, index, am)
                    self.children[index].append(self.indexToNow)
        else:
            for kp in list(set(self.keypoints.keys()) - set([0])):
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = kp
                self.indexToActionID[self.indexToNow] = 0
                self.initialiseLeafNode(self.indexToNow, index, {})
                self.children[index].append(self.indexToNow)

        self.fullyExpanded[index] = True
        self.usedActionsID = {}
        return self.children[index]

    def backPropagation(self, index, value):
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent:
            print("start backPropagating the value %s from node %s, whose parent node is %s" % (
                value, index, self.parent[index]))
            self.backPropagation(self.parent[index], value)
        else:
            print("backPropagating ends on node %s" % index)

    # start random sampling and return the Euclidean value as the value
    def sampling(self, index, availableActions):
        print("start sampling node %s" % index)
        availableActions2 = copy.deepcopy(availableActions)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples):
            self.atomicManipulationPath = self.manipulation[index]
            self.depth = 0
            self.availableActionIDs = {}
            for k in self.keypoints.keys():
                self.availableActionIDs[k] = list(availableActions2[k].keys())
            self.usedActionIDs = {}
            for k in self.keypoints.keys():
                self.usedActionIDs[k] = []
            (childTerminated, val) = self.sampleNext(self.keypoint[index])
            self.numOfSampling += 1
            sampleValues.append(val)
            i += 1
        return childTerminated, min(sampleValues)

    def computeDistance(self, newImage):
        (distMethod, _) = self.eta
        return distMethod.dist(newImage, self.image)

    def sampleNext(self, k):
        activations1 = self.moves.applyManipulation(self.image, self.atomicManipulationPath)
        (newClass, newConfident) = self.model.predict(activations1)

        (distMethod, distVal) = self.eta
        dist = self.computeDistance(activations1)

        # need not only class change, but also high confidence adversary examples
        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
            print("sampling a path ends in a terminal node with depth %s... " % self.depth)
            self.atomicManipulationPath = self.scrutinizePath(self.atomicManipulationPath)
            self.numAdv += 1
            print("current best %s, considered to be replaced by %s" % (self.bestCase[0], dist))
            if self.bestCase[0] > dist:
                print("update best case from %s to %s" % (self.bestCase[0], dist))
                self.numConverge += 1
                self.bestCase = (dist, self.atomicManipulationPath)
                path0 = "%s_pic/%s_Unsafe_currentBest_%s.png" % (self.data_set, self.image_index, self.numConverge)
                self.model.save_input(activations1, path0)
            return (self.depth == 0, dist)

        elif dist > distVal:   ##########################
            print("sampling a path ends by eta with depth %s ... " % self.depth)
            return (self.depth == 0, distVal)

        elif not list(set(self.availableActionIDs[k]) - set(self.usedActionIDs[k])): ####################
            print("sampling a path ends with depth %s because no more actions can be taken ... " % self.depth)
            return (self.depth == 0, distVal)

        # elif self.depth > (self.eta[1] / self.tau) * 2:
        #    print(
        #        "sampling a path ends with depth %s more than the prespecifided maximum sampling depth ...  the largest distance is %s " % (self.depth,dist) )
        #    return (self.depth == 0, distVal)

        else:
            # print("continue sampling node ... ")
            # randomActionIndex = random.choice(list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k])))

            i = 0
            while True:

                randomActionIndex = random.choice(self.availableActionIDs[k])
                if k == 0:
                    nextAtomicManipulation = {}
                else:
                    nextAtomicManipulation = self.actions[k][randomActionIndex]

                if self.usefulAction(self.atomicManipulationPath,
                                     nextAtomicManipulation) == True or nextAtomicManipulation == {} or i > 10:
                    break

                i += 1

                # self.availableActionIDs[k].remove(randomActionIndex)
                # self.usedActionIDs[k].append(randomActionIndex)
            newManipulationPath = mergeTwoDicts(self.atomicManipulationPath, nextAtomicManipulation)
            activations2 = self.moves.applyManipulation(self.image, newManipulationPath)
            (newClass2, newConfident2) = self.model.predict(activations2)

            self.atomicManipulationPath = newManipulationPath
            self.depth = self.depth + 1
            if k == 0:
                return self.sampleNext(randomActionIndex)
            else:
                return self.sampleNext(0)

    def scrutinizePath(self, manipulations):
        flag = False
        tempManipulations = copy.deepcopy(manipulations)
        for k, v in manipulations.items():
            tempManipulations[k] = 0
            activations1 = self.moves.applyManipulation(self.image, tempManipulations)
            (newClass, newConfident) = self.model.predict(activations1)
            if newClass != self.originalClass:
                manipulations.pop(k)
                flag = True
                break

        if flag is True:
            return self.scrutinizePath(manipulations)
        else:
            return manipulations
