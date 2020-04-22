from __future__ import print_function
from NeuralNetwork import *
from DataSet import *
from CompetitiveMCTS import *
from CooperativeMCTS import *
from bound import *


class upperbound(bound):
    """Main search function
    Return format: tuple of
            delta timestamp: total running time
          float [0, 1]: new confidence in image
          float [0, 100]: percentage modified
          float: new L2 distance
          float: new L1 distance
          float: new L0 distance
          int: maximally vulnerable feature (value can be any if cooperative)"""
    def search(self, image_index):
        image = self.dataset.get_input(image_index)
        label, newConfident = self.nn.predict(image)
        label_str = self.nn.get_label(int(label))

        if self.verbose == 1:
            print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, label_str, conf))
            print("the second player is %s." % gameType)

        if self.gameType == "cooperative":
            mcts = MCTSCooperative(data_set = self.data_set_name,
                                model = self.nn,
                                image_index = image_index,
                                image = image,
                                tau = self.tau,
                                eta = self.eta,
                                attention = self.attention)
        else:
            mcts = MCTSCompetitive(data_set = self.data_set_name,
                                model = self.nn,
                                image_index = image_index,
                                image = image,
                                tau = self.tau,
                                eta = self.eta,
                                attention = self.attention)

        # Expand search tree until timeout
        mcts = self.search_inner_loop(mcts)

        _, best_manip = mcts.best_case        

        print("The number of sampling is %s" % mcts.numOfSampling)
        print("The number of adversarial examples found: %s\n" % mctsInstance.numAdv)

        adversary = mcts.applyManipulation(best_manip)
        new_label, new_conf = self.nn.predict(image1)
        new_label_str = self.nn.get_label(int(new_label))

        # If adversarial example found, output the best results
        if new_class != label and best_value < eta[1]:
            l2dist, l1dist, l0dist, percent = self.success_callback(image_index, adversary)
            return time.time() - start_time_all, new_conf, percent, l2dist, l1dist, l0dist, 0
        else:
            self.fail_callback()
            return 0, 0, 0, 0, 0, 0, 0

    # Expand the MCTS tree until timeout
    def search_inner_loop(self, mcts):
        mcts.initialiseMoves()

        start_time_all = time.time()
        running_time_all = 0

        start_time_level = time.time()
        running_time_level = 0

        # Start with best distance being the search cap
        current_best = self.eta[1]
        current_best_index = 0

        # Interrupt search when total running time exceeds cap
        while running_time_all <= self.MCTS_all_maximal_time:
            # Three steps for MCTS
            (leafNode, availableActions) = mcts.treeTraversal(mcts.rootIndex)
            newNodes = mcts.initialiseExplorationNode(leafNode, availableActions)
            for node in newNodes:
                _, value = mcts.sampling(node, availableActions)
                mcts.backPropagation(node, value)
            if current_best > mcts.bestCase[0]:
                if verbose == 1:
                    print("Best distance up to now is %s" % (str(mcts.bestCase[0])))
                current_best = mcts.bestCase[0]
                current_best_index += 1

            best_child = mcts.bestChild(mctsInstance.rootIndex)

            # Store the current best
            _, best_manip = mctsInstance.bestCase
            image1 = mcts.applyManipulation(best_manip)
            path0 = "%s_pic/%s_Unsafe_currentBest_%s.png" % (self.data_set_name, image_index, current_best_index)
            self.nn.save_input(image1, path0)

            running_time_all = time.time() - start_time_all
            running_time_level = time.time() - start_time_level

        return mcts

    def failure_callback(self):
        if self.game_type == "cooperative":
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")

        else 
            print("\nthe robustness of the (input, model) is under control, "
                      "with the first player is able to defeat the second player "
                      "who aims to find adversarial example by "
                      "playing suitable strategies on selecting features. ")