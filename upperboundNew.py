
from __future__ import print_function
from NeuralNetwork import *
from DataSet import *
from MCTS import *
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
        self.bound_type = 'ub'


        image = self.dataSet.get_input(image_index)
        label, conf = self.model.predict(image)
        label_str = self.model.get_label(int(label))

        if self.verbose == 1:
            print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, label_str, conf))
            print("the second player is %s." % self.game_type)


        mcts = MCTS(data_set_name = self.data_set_name,
                    model = self.model,
                    image_index = image_index,
                    image = image,
                    tau = self.tau,
                    eta = self.eta,
                    attention = self.attention)

        if self.game_type == "cooperative":
            mcts.set_root(NodeCooperative())
        else:
            mcts.set_root(NodeCompetitive())

        # Expand search tree until timeout
        start_time_all = time.time()
        mcts.search(300)

        best_manip, best_value = mcts.best_case        

        print("The number of iterations is %s" % mcts.iter_count)
        print("The number of adversarial examples found: %s\n" % mcts.num_adv)

        adversary = mcts.moves.applyManipulation(image, best_manip)
        new_label, new_conf = self.model.predict(adversary)
        new_label_str = self.model.get_label(int(new_label))

        # If adversarial example found, output the best results
        if new_label != label and best_value < self.eta[1]:
            l2dist, l1dist, l0dist, percent = self.success_callback(image_index, adversary)
            if self.game_type == 'competitive':
                feat = mcts.most_vulnerable_feature
                img_feat = image
                for (a, b, c), dist in mcts.moves.player_two[feat].items():
                    img_feat[a, b] = [1, 0, 0]
                path0 = "%s_pic/%s_%s/%s_feature.png" % (
                    self.data_set_name,
                    self.bound_type,
                    self.game_type,
                    image_index
                )
                self.model.save_input(img_feat, path0)
            return time.time() - start_time_all, new_conf, percent, l2dist, l1dist, l0dist, 0
        else:
            self.failure_callback()
            return 0, 0, 0, 0, 0, 0, 0


    def failure_callback(self):
        if self.game_type == "cooperative":
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")

        else:
            print("\nthe robustness of the (input, model) is under control, "
                      "with the first player is able to defeat the second player "
                      "who aims to find adversarial example by "
                      "playing suitable strategies on selecting features. ")