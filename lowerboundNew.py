#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a 'lowerbound' class (child of 'bound') to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative, or Player I's maximum
adversary distance whilst Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk

Refactored: Denitsa Markova
Email: denitsa.markova@stcatz.ox.ac.uk
"""

from CooperativeAStarNew import *
from CompetitiveAlphaBeta import *
from NeuralNetwork import *
from DataSet import *
from bound import *

class lowerbound(bound):

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
        image = self.data_set.get_input(image_index)
        label, confidence = self.nn.predict(image)
        label_str = self.nn.get_label(int(label))

        if self.verbose == 1:
            print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, label_str, confidence))
            print("The second player is being %s." % game_type)

        path = "%s_pic/idx_%s_label_[%s]_with_confidence_%s.png" % (
        dataset_name, image_index, label_str, confidence)
        NN.save_input(image, path)

        start_time = time.time()

        if self.game_type == 'cooperative':
            search_instance = CooperativeAStar(data_set_name = self.data_set_name,
                                    image_index = image_index,
                                    image = image,
                                    model = self.nn,
                                    eta = self.eta,
                                    tau = self.tau,
                                    attention = self.attention)
        else:
            search_instance = CompetitiveAlphaBeta(image = image,
                                            model = slef.nn,
                                            eta = self.eta,
                                            tau = self.tau,
                                            attention = self.attention)
        
        search_instance.play_game()
        if search_instance.ADVERSARY_FOUND is True:
            adversary = search_instance.ADVERSARY
            new_label, new_conf = self.nn.predict(image1)
            new_label_str = self.nn.get_label(int(new_label))

            self.success_callback(search_instance.ADVERSARY)
            return time.time() - start_time, new_conf, percent, l2dist, l1dist, l0dist, 0
        else:
            self.failure_callback()
            return 0, 0, 0, 0, 0, 0, 0

    def failure_callback(self):
        print("Adversarial distance exceeds distance budget.")
