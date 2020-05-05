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

from FeatureExtraction import *
from basics import *
from GameMoves import *

import time

THETA = 0.5

class GameNode:
    def __init__(self, state, image, path_cost, heuristic, parent, depth):
        self.state = state
        self.image = image
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.parent = parent
        self.depth = depth

    def value(self):
        return self.path_cost + self.heuristic

class AStarSearch:
    def __init__(self, initial_image, goal_test, heuristic_function, actions, cost_function, apply_action_function, verbose = 1):
        self.initial_image = initial_image
        self.goal_test = goal_test
        self.heuristic_function = heuristic_function
        self.actions = actions
        self.cost_function = cost_function
        self.apply = apply_action_function
        self.verbose =verbose

        h = self.heuristic_function(initial_image)

        self.root = GameNode(state = {},
                            image = self.initial_image,
                            path_cost = 0,
                            heuristic = h,
                            parent = None,
                            depth = 0)

        self.search_queue = []
        self.visited = set()

        self.best_case = (0, None)
        self.best_estimate = 0

        # Using the confidence as a tie breaker
        heapq.heappush(self.search_queue, (self.root.value(),  10000*self.root.heuristic, self.root))

    def search(self, tau):
        n_searched = 0

        while len(self.search_queue) > 0:
            value, tiebreak, search_node = heapq.heappop(self.search_queue)
            if self.flat(search_node.state) in self.visited:
                continue

            self.visited.add(self.flat(search_node.state))

            n_searched += 1
            true_cost = self.cost_function(self.initial_image, search_node.image)
            if search_node.parent is not None and true_cost == 0:
                continue 

            # best_estimate is the best bound we have through the heuristic
            # Only use if heuristic and tau are true values
            self.best_estimate = value
            if true_cost > self.best_case[0]:
                self.best_case = (true_cost, search_node.state)

            if self.verbose == 1:
                print("State being considered at depth %d" % search_node.depth)
                print("Estimated distance: %f" % value)
                print("Tiebreaker value: %f" % tiebreak)
                print("True distance: %f" % self.cost_function(self.initial_image, search_node.image))
                print("Path: %s" % search_node.state)
                print("%d items searched so far" % n_searched)
                print("%d items in search queue" % len(self.search_queue))
                print("%d different states visited" % len(self.visited))


            if self.goal_test(search_node.image) or search_node.heuristic < THETA * tau:
                return self.best_case

            eta_pruned = 0

            for action in self.actions:
                res = self.apply(search_node.state, action)

                if res is None:
                    eta_pruned += 1
                    continue

                next_image, next_state = res

                if next_state is not None and self.flat(next_state) not in self.visited:
                    cost = self.cost_function(search_node.image, next_image)

                    if cost == 0:
                        continue

                    h = self.heuristic_function(next_image)
                    child_node = GameNode(image = next_image,
                            state = next_state,
                            path_cost = search_node.path_cost + cost,
                            heuristic = h,
                            parent = search_node,
                            depth = search_node.depth + 1
                        )


                    heapq.heappush(self.search_queue, (child_node.value(), 10000*child_node.heuristic, child_node))
            print("%d  child nodes out of search bound." % eta_pruned)


        return self.best_case


    def flat(self, state):
        return tuple(sorted(state.items()))


class CooperativeAStar:
    def __init__(self, data_set_name, image_index, image, model, eta, tau, attention = False):
        self.data_set_name = data_set_name
        self.image_index = image_index
        self.image = image
        self.model = model
        self.eta = eta
        self.tau = tau
        self.attention = attention
        self.root_label, self.root_conf = self.model.predict(self.image)

        self.game_moves = GameMoves(model = self.model,
                                    image = self.image,
                                    tau = self.tau,
                                    pixel_bounds = (0,1),
                                    attention = self.attention)
        self.game_moves.generate_moves(collapse_channels=False)

        self.dist_evaluation = {}
        self.adv_manip = ()
        self.adversary_found = None
        self.adversary = None

        self.current_safe = [0]

    def goal_test(self, image):
        new_label, new_confidence = self.model.predict(image)
        return new_label != self.root_label

    def heuristic_function(self, image):
        probabilities = self.model.model.predict(np.array([image]))[0]
        p_max, p_2nd_max = heapq.nlargest(2, probabilities)
        # Assuming L = 1000, to fix? 

        heuristic = (p_max - p_2nd_max) * self.tau
        return heuristic

    def apply_action_function(self, manip, action):
        new_manip = mergeTwoDicts(manip, action)
        next_image= self.game_moves.applyManipulation(self.image, new_manip)

        # if valid is false
        if self.eta[0].dist(self.image, next_image) > self.eta[1]:
            return None

        return next_image, new_manip


    def search(self, image):
        new_image = copy.deepcopy(self.image)
        new_label, new_confidence = self.model.predict(new_image)

        astar = AStarSearch(
                initial_image = new_image,
                goal_test = self.goal_test,
                heuristic_function = self.heuristic_function,
                actions = [move for key, partition in self.game_moves.player_two.items() if key > 0 for move in partition],
                cost_function = self.eta[0].dist,
                apply_action_function = self.apply_action_function
            )

        self.best_case = astar.search(self.tau)

        # TODO add safe images