"""
Construct a class to handle A* search on our game tree

Author: 1013952
"""

import heapq

from FeatureExtraction import *
from basics import *
from GameMoves import *

import time

THETA = 0.5

# Nodes of the game tree - store the state (manipulation) and values needed for A*
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


# AStar class, used to run searches on the game tree
class AStarSearch:
    def __init__(self, initial_image, goal_test, heuristic_function, actions, 
                    cost_function, apply_action_function, verbose = 1):
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
        heapq.heappush(self.search_queue, (self.root.value(),  self.root.heuristic, self.root))

    # Returns a tuple of (best manipulation, best value)
    def search(self, tau):
        n_searched = 0

        while len(self.search_queue) > 0:
            value, tiebreak, search_node = heapq.heappop(self.search_queue)
            if self.flat(search_node.state) in self.visited:
                continue

            self.visited.add(self.flat(search_node.state))

            n_searched += 1
            true_cost = self.cost_function(search_node.state)#self.initial_image, search_node.image)
            if search_node.parent is not None and true_cost == 0:
                continue 

            # best_estimate is the best bound we have through the heuristic
            # Only use if heuristic and tau are true values
            self.best_estimate = value
            if true_cost > self.best_case[0]:
                self.best_case = (search_node.state, true_cost)

            if self.verbose == 1:
                print("State being considered at depth %d" % search_node.depth)
                print("Estimated distance: %f" % value)
                print("Tiebreaker value: %f" % (search_node.heuristic / tau / 28))
                print("True distance: %f" % true_cost) #self.initial_image, search_node.image))
                print("Path: %s" % search_node.state)
                print("%d items searched so far" % n_searched)
                print("%d items in search queue" % len(self.search_queue))
                print("%d different states visited" % len(self.visited))


            if self.goal_test(search_node.image) or search_node.heuristic < THETA * tau * 28:
                return self.best_case

            eta_pruned = 0

            for action in self.actions:
                res = self.apply(search_node.state, action)

                if res is None:
                    eta_pruned += 1
                    continue

                next_image, next_state = res

                if next_state is not None and self.flat(next_state) not in self.visited:
                    cost = self.cost_function(next_state) #self.initial_image, next_image)

                    if cost == 0 or  cost <= search_node.path_cost:
                        continue

                    h = self.heuristic_function(next_image)
                    child_node = GameNode(image = next_image,
                            state = next_state,
                            path_cost = cost,
                            heuristic = h,
                            parent = search_node,
                            depth = search_node.depth + 1
                        )


                    heapq.heappush(self.search_queue, (child_node.value(), 10000*child_node.heuristic, child_node))
            print("%d  child nodes out of search bound." % eta_pruned)


        return self.best_case


    # State is a dictionary - cannot be put into a set without flattening
    def flat(self, state):
        return tuple(sorted(state.items()))

# Interfaes between the core A* search and the specifics of our search space
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
        self.game_moves.generate_moves(collapse_channels= False)

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

        heuristic = (p_max - p_2nd_max) * self.tau * 28
        return heuristic

    def apply_action_function(self, manip, action):
        new_manip = mergeTwoDicts(manip, action)
        next_image= self.game_moves.applyManipulation(self.image, new_manip)

        # if valid is false
        if self.eta[0].dist(self.image, next_image) > self.eta[1]:
            return None

        return next_image, new_manip

    def cost_fun(self, manip):
        sum = 0
        for loc, value in manip.items():
            sum += min(1, value * value)
        return np.sqrt(sum)

    # Can set k to a value in range(number of moves for player A) to restrict
    # search to just that move - for competitive
    def search(self, image, k = None):
        new_image = copy.deepcopy(self.image)
        new_label, new_confidence = self.model.predict(new_image)

        if k is None:
            astar = AStarSearch(
                    initial_image = new_image,
                    goal_test = self.goal_test,
                    heuristic_function = self.heuristic_function,
                    actions = [move for key, partition in self.game_moves.player_two.items() if key > 0 for move in partition],
                    cost_function = self.cost_fun,
                    apply_action_function = self.apply_action_function
                )
        else:
            print("%d actions loaded." % len(self.game_moves.player_two[k]))
            astar = AStarSearch(
                    initial_image = new_image,
                    goal_test = self.goal_test,
                    heuristic_function = self.heuristic_function,
                    actions = self.game_moves.player_two[k],
                    cost_function = self.cost_fun,
                    apply_action_function = self.apply_action_function
                )


        self.best_case = astar.search(self.tau)
