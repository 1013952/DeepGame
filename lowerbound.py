#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a 'lowerbound' function to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative, or Player I's maximum
adversary distance whilst Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

from CooperativeAStar import *
from CompetitiveAlphaBetaNew import *
from NeuralNetwork import *
from DataSet import *
from bound import *

class lowerbound(bound):


    def search(self, image_index):
        self.bound_type = 'lb'

        image = self.dataSet.get_input(image_index)
        label, conf = self.model.predict(image)
        label_str = self.model.get_label(int(label))

        if self.verbose == 1:
            print("Working on input with index %s, whose class is '%s' and the confidence is %s."
                  % (image_index, label_str, conf))
            print("The second player is being %s." % self.game_type)

        # path = "%s_pic/idx_%s_label_[%s]_with_confidence_%s.png" % (
        #     self.data_set_name, image_index, label_str, confidence)
        # self.save_input(image, path)

        if self.game_type == 'cooperative':
            game_instance = CooperativeAStar(data_set_name = self.data_set_name,
                                            image_index = image_index,
                                            image = image,
                                            model = self.model,
                                            eta = self.eta,
                                            tau = self.tau,
                                            attention = self.attention)
        else:
            game_instance = CompetitiveAlphaBeta(data_set_name = self.data_set_name,
                                                image_index = image_index,
                                                image = image,
                                                model = self.model,
                                                eta = self.eta,
                                                tau = self.tau,
                                                attention = self.attention)

        start_time = time.time()
        game_instance.search(image_index)

        best_manip, best_value = game_instance.best_case
        adversary = game_instance.game_moves.applyManipulation(image, best_manip)
        new_label, new_conf = self.model.predict(adversary)
        new_label_str = self.model.get_label(int(new_label))

        if new_label != label:
            l2dist, l1dist, l0dist, percent = self.success_callback(image_index, adversary)
            return time.time() - start_time, new_conf, percent, l2dist, l1dist, l0dist, 0
        else:
            self.failure_callback()
            return 0, 0, 0, 0, 0, 0, 0



        def failure_callback(self):
            print("Adversarial distance exceeds distance budget.")
