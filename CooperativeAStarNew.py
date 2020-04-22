#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a CooperativeAStar class to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import heapq

from FeatureExtractionNew import *
from basicsNew import *
from GameMovesNew import *

import time

class GameNode:
    def __init__(self, state, path_cost, heuristic, parent, depth, a, b):
        self.state = state
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.parent = parent
        self.depth = depth
        self.p = (a, b)

    def value(self):
        return self.path_cost + self.heuristic

class AStarSearch:
    def __init__(self, initial_state, goal_test, heuristic_function, actions, cost_function, apply_action_function, verbose = 1):
        self.initial_state = initial_state
        self.goal_test = goal_test
        self.heuristic_function = heuristic_function
        self.actions = actions
        self.cost_function = cost_function
        self.apply = apply_action_function
        self.verbose =verbose

        h, a, b = self.heuristic_function(initial_state)

        self.root = GameNode(state = self.initial_state,
                            path_cost = 0,
                            heuristic = h,
                            parent = None,
                            depth = 0, 
                            a = a,
                            b = b)

        self.search_queue = []
        self.visited = {tuple(self.initial_state.flatten())}
        heapq.heappush(self.search_queue, (self.root.value(), self.root))

    def search(self):
        n_searched = 0

        while len(self.search_queue) > 0:

            value, search_node = heapq.heappop(self.search_queue)
            n_searched += 1

            if self.verbose == 1:
                print("State being considered at depth %d" % search_node.depth)
                print("Estimated distance: %f" % value)
                print("True distance: %f" % self.cost_function(self.initial_state, search_node.state))
                print("%d items searched so far" % n_searched)
                print("%d items in search queue" % len(self.search_queue))


            if self.goal_test(search_node.state):
                return search_node

            for action in self.actions:
                next_state = self.apply(search_node.state, action)

                if next_state is not None and tuple(next_state.flatten()) not in self.visited:
                    self.visited.add(tuple(next_state.flatten()))

                    cost = self.cost_function(search_node.state, next_state)

                    h, p_max, p_2nd_max = self.heuristic_function(next_state)
                    child_node = GameNode(
                            state = next_state,
                            path_cost = search_node.path_cost + cost,
                            heuristic = h,
                            parent = search_node,
                            depth = search_node.depth + 1,
                            a = p_max,
                            b = p_2nd_max
                        )
                    heapq.heappush(self.search_queue, (child_node.value(), child_node))

        return None


class CooperativeAStar:
    def __init__(self, dataset, idx, image, model, eta, tau, bounds=(0, 1), attention = False):
        self.DATASET = dataset
        self.IDX = idx
        self.IMAGE = image
        self.IMAGE_BOUNDS = bounds
        self.MODEL = model
        self.DIST_METRIC = eta[0]
        self.DIST_VAL = eta[1]
        self.TAU = tau
        self.ATTENTION = attention
        self.LABEL, _ = self.MODEL.predict(self.IMAGE)
        self.GAME_MOVES = GameMoves(model = self.MODEL,
                                    image = self.IMAGE,
                                    tau = self.TAU,
                                    pixel_bounds = self.IMAGE_BOUNDS,
                                    attention = self.ATTENTION)

        self.DIST_EVALUATION = {}
        self.ADV_MANIPULATION = ()
        self.ADVERSARY_FOUND = None
        self.ADVERSARY = None

        self.CURRENT_SAFE = [0]

        print("Distance metric %s, with bound value %s." % (self.DIST_METRIC, self.DIST_VAL))

    def goal_test(self, image):
        new_label, new_confidence = self.MODEL.predict(image)
        return new_label != self.LABEL

    def heuristic_function(self, image):
        probabilities = self.MODEL.model.predict(np.array([image]))[0]
        p_max, p_2nd_max = heapq.nlargest(2, probabilities)
        heuristic = (p_max - p_2nd_max) * 2 * self.TAU
        return heuristic, p_max, p_2nd_max

    def apply_action_function(self, image, action):
        valid, next_image= self.GAME_MOVES.applyManipulation(image, action, report_valid=True)

        # if valid is false
        if self.DIST_METRIC.dist(self.IMAGE, next_image) > self.DIST_VAL:
            return None

        return next_image


    def play_game(self, image):
        new_image = copy.deepcopy(self.IMAGE)
        new_label, new_confidence = self.MODEL.predict(new_image)

        astar = AStarSearch(
                initial_state = new_image,
                goal_test = self.goal_test,
                heuristic_function = self.heuristic_function,
                actions = [move for key, partition in self.GAME_MOVES.moves.items() if key > 0 for move in partition],
                cost_function = self.DIST_METRIC.dist,
                apply_action_function = self.apply_action_function
            )

        result = astar.search()
        if result is None:
            self.ADVERSARY_FOUND = False
        else:
            self.ADVERSARY_FOUND = True
            self.ADVERSARY = new_image

        # TODO add safe images