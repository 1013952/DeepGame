from AttentionNetwork import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0


nn = AttentionNetwork()

nn.train_network(data_set_name = 'mnist')

nn.save_network()
