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

class Node:
    # Trick for variable sharing
    game_moves = [None]
    next_index = [0]

    def __init__(self, manipulation, depth = 0, children = None, max_player = None, player_one_choice = None):
        self.depth = depth
        self.index = self.next_index[0]
        self.next_index[0] += 1
        self.manipulation = manipulation
        if children is None:
            self.children = []
        else:
            self.children = children
        self.max_player = max_player
        self.player_one_choice = player_one_choice

    def flat(self):
        return tuple(sorted(self.manipulation.items()))

class PlayerOneNode(Node):
    def make_children(self):
        self.children = []
        for move in self.game_moves[0].player_one:
            self.children.append(PlayerTwoNode(
                    depth = self.depth,
                    manipulation = self.manipulation,
                    children = None,
                    max_player = False,
                    player_one_choice = move
                ))

class PlayerTwoNode(Node):
    def make_children(self):
        self.children = []
        for move in self.game_moves[0].player_two[self.player_one_choice]:
            new_manip = mergeTwoDicts(self.manipulation, move)
            if len(new_manip) <= len(self.manipulation):
                continue
            self.children.append(PlayerOneNode(
                depth = self.depth + 1,
                manipulation = mergeTwoDicts(self.manipulation, move),
                children = None,
                max_player = True,
                player_one_choice = None))




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
        self.depth_cutoff = 50
        self.best_case = (self.eta[1], {})

        self.visited = set()

        self.robust_feature_found = False
        self.robust_feature = []
        self.fragile_feature_found = False
        self.least_fragile_feature = []

    def alphabeta(self, alpha, beta, node):
        node_image = self.game_moves.applyManipulation(self.image, node.manipulation)
        dist = self.eta[0].dist(self.image, node_image)

        if dist > self.eta[1]:
            return self.eta[1]

        if node.depth > self.depth_cutoff:
            return self.eta[1] # Change to heuristic - lower-bound from L1 dist

        self.visited.add(node.flat())
        if len(self.visited) % 100 == 0:
            print("%d nodes visited." % len(self.visited))

        new_label, new_conf = self.model.predict(node_image)
        if new_label != self.root_label:
            print("Adversarial example found at distance %d." % d)
            return dist 

        if node.max_player is True:
            value = - np.infty
            node.make_children()
            for child in node.children:
                if child in self.visited:
                    continue

                value = max(value, self.alphabeta(alpha, beta, child))
                if value > alpha:
                    self.best_case = (node.manipulation, value)
                    if node.depth < 3:
                        print("Alpha updating to %f at node %d." % (value, node.index))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break # beta cutoff
            return value
        else:
            value = np.infty
            node.make_children()
            for child in node.children:
                if child in self.visited:
                    continue 
                child_image = self.game_moves.applyManipulation(self.image, child.manipulation)
                if self.flat(child_image) in self.visited:
                    continue
                value = min(value, self.alphabeta(alpha, beta, child))
                if value < beta:
                    if node.depth < 3:
                        print("Beta updating to %f at node %d." % (value, node.index))
                beta = min(beta, value)
                if alpha >= beta:
                    break # alpha cutoff
            return value

    def flat(self, image):
        return tuple(image.flatten())


    def search(self, image_index):
        self.visited = set()
        root_node = PlayerOneNode(manipulation = {})
        root_node.game_moves[0] = self.game_moves

        return self.alphabeta(alpha = -np.inf, beta = np.inf, node = root_node)