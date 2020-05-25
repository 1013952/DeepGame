"""
Implementation of Monte Carlo Tree Search
For our use case
Author: 1013952
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

parallel_pool_size = 10


# Game node - handles whose move it is, encapsulates generation of children
class Node:
    def __init__(self, index = 0, manipulation = None,
                player_to_act = 1, value = 0, depth = 0,
                parent = None, children = None, attempts = 0,
                player_one_choice = None, mult = 1):
        self.index = index
        if manipulation is None:
            self.manipulation = {}
        else:
            self.manipulation = manipulation
        self.value = value
        self.depth = depth
        self.parent = parent
        if children is None:
            self.children = []
        self.attempts = attempts

        assert player_to_act in [1, 2]
        self.player_to_act = player_to_act
        self.player_one_choice = player_one_choice


    def random_child(self):
        NodeType = self.__class__
        if self.player_to_act == 1:
            next_move = np.random.choice(self.game_moves[0].player_one)
            child = NodeType(index = -1,
                        manipulation = self.manipulation,
                        player_to_act = 2,
                        depth = self.depth + 1,
                        parent = self,
                        player_one_choice = next_move)
        else:
            next_move = np.random.choice(self.game_moves[0].player_two[self.player_one_choice])
            child = NodeType(index = -1,
                        manipulation = mergeTwoDicts(self.manipulation, next_move),
                        player_to_act = 1,
                        depth = self.depth + 1,
                        parent = self)

        return child

    # Function to expand all children of a certain node
    # Returns first next usable index
    def create_children(self):
        if len(self.children) > 0:
            return
        NodeType = self.__class__
        if self.player_to_act == 1:
            for move in self.game_moves[0].player_one:
                self.children.append(NodeType(index = self.next_index[0],
                            manipulation = self.manipulation,
                            player_to_act = 2,
                            depth = self.depth + 1,
                            parent = self,
                            player_one_choice = move))
                self.next_index[0] += 1
        else:
            for move in self.game_moves[0].player_two[self.player_one_choice]:
                self.children.append(NodeType(index = self.next_index[0],
                            manipulation = mergeTwoDicts(self.manipulation, move),
                            player_to_act = 1,
                            depth = self.depth + 1,
                            parent = self))
                self.next_index[0] += 1

    def get_value(self):
        if self.attempts == 0:
            return 0
        else:
            return self.value / self.attempts


# Specifics for cooperative
class NodeCooperative(Node):
    # Trick for sharing variables between instances
    next_index = [1]
    game_moves = [None]

# Specifics for competitive
class NodeCompetitive(Node):
    # Trick for sharing variables between instances
    next_index = [1]
    game_moves = [None]

    # Value is slightly different
    def get_value(self):
        if self.attempts == 0:
            return 0
        elif self.player_to_act == 1:
            return - self.value / self.attempts
        else:
            return self.value / self.attempts



# Conducts MCTS 
class MCTS:
    def __init__(self, data_set_name, model, image_index, image, tau, eta, attention=False, verbose = 1):
        self.data_set_name = data_set_name
        self.image_index = image_index
        self.image = image
        self.model = model
        self.tau = tau
        self.eta = [eta[0], eta[1]]
        self.attention = attention

        self.root_class, self.root_conf = self.model.predict(self.image)

        self.moves = GameMoves(model = self.model,
                                image = self.image,
                                tau = self.tau,
                                pixel_bounds = (0, 1),
                                attention = self.attention,
                                verbose = 1)
        self.moves.generate_moves(collapse_channels = True)


        self.best_case = ({}, self.eta[1])

        # For tracking intermediate images
        self.num_converge = 0
        self.iter_count = 0
        self.num_adv = 0


    def set_root(self, root):
        self.root_node = root
        self.root_node.game_moves[0] = self.moves

        self.root_node.create_children()




    def search(self, time_cap):
        start_time = time.time()
        while time.time() - start_time < time_cap:
            new_value = self.expansion(self.selection(self.root_node))
            print("Iteration %d terminated with value %f." %
                    (self.iter_count, self.best_case[1]))
            print("Time elapsed: %f out of %f." %
                    (time.time() - start_time, time_cap))
            self.iter_count += 1


    def selection(self, node):
        # Explore down the tree until we reach a leaf
        # print("Exploring from node %s with children %s." % (node.index, [child.index for child in node.children]))
        if len(node.children) > 0:
            unexplored = [child for child in node.children if child.attempts == 0]
            if len(unexplored) > 0:
                return unexplored[0]

            # Standard UCT formula
            values = [child.get_value()
                                         + explorationRate * math.sqrt(
                            math.log(node.attempts) / child.attempts)
                            for child in node.children]
            values = np.array(values)

            return self.selection(np.random.choice(node.children, p = values/sum(values)))
        
        print("Returning node %d." % node.index)
        return node


    # Expand a node
    def expansion(self, node):
        print("Expanding node %d with root probabilities %s." % 
            (node.index, [child.value / max(0.001, child.attempts) for child in self.root_node.children]))
        # Check for termination conditions
        res = self.check_termination(node)
        if res is not None:
            self.back_prop(node, res)
            return res

        # Play out uniformly at random from 
        node.create_children()

        for i in range(MCTS_multi_samples):
            next_child = np.random.choice(node.children)
            res = self.simulation(next_child)
            self.back_prop(node, res)
        return res


    def simulation(self, node):
        if node.index > -1:
            print("Simulating unexplored node %d at depth %d." % (node.index, node.depth))
        term_condition = self.check_termination(node)
        if term_condition is None:
            return self.simulation(node.random_child())
        return term_condition


    # Propagate value up the tree
    def back_prop(self, node, value):
        if node.parent is None:
            print("Back-propagating value %f." % value)
        node.value += value
        node.attempts += 1

        if node.parent is not None:
            self.back_prop(node.parent, value)


    # Checks termination conditions:
    # Returns value if terminating
    #         None if not
    def check_termination(self, node):
        node_image = self.moves.applyManipulation(self.image, node.manipulation)
        new_class, new_conf = self.model.predict(node_image)
        node.conf = new_conf

        # Adversarial example found
        if new_class != self.root_class:
            print("Adversarial example found in node %d at depth %d." % (
                    node.index, node.depth)
                )
            node.manipulation = self.clean_path(node.manipulation)
            node_image = self.moves.applyManipulation(self.image, node.manipulation)

            self.num_adv += 1
            dist = self.eta[0].dist(self.image, node_image)
            self.back_prop(node, dist)

            if dist < self.best_case[1]:
                print("Updating best case value from %f to %f." %
                    (self.best_case[1], dist))
                self.best_case = (node.manipulation, dist)
                if node.player_one_choice is None:
                    self.most_vulnerable_feature = node.parent.player_one_choice
                else:
                    self.most_vulnerable_feature = node.player_one_choice

                self.num_converge += 1
                # Save intermediate images
                path0 = "%s_pic/%s_Unsafe_currentBest_%s.png" % (
                            self.data_set_name,
                            self.image_index,
                            self.num_converge)
                self.model.save_input(node_image, path0)
            return dist

        # Termination by eta
        if self.eta[0].dist(self.image, node_image) > self.eta[1]:
            print("Termination by eta in node %d at depth %d." % (
                    node.index, node.depth))
            self.back_prop(node, self.eta[1])
            return self.eta[1]

        return None



    # Cull unnecessary manipulations on path
    def clean_path(self, manipulation):
        flag = False
        copy_manip = copy.deepcopy(manipulation)

        for k, v in manipulation.items():
            copy_manip.pop(k, None)

            new_image = self.moves.applyManipulation(self.image, copy_manip)
            new_class, new_conf = self.model.predict(new_image)

            # Add atomic manipulation if it is crucial for change
            if new_class == self.root_class:
                copy_manip[k] = v

        return copy_manip