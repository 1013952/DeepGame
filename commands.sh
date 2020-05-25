#!/bin/bash
for i in {831..835}
do
	python main.py cifar10 lb cooperative $i L2 0.1 0.001 y 2>&1 | tee log_cifar_lb_cooperative_L2_$i.txt
	# python main.py gtsrb ub cooperative $i L0 10 1 y 2>&1 | tee log_gtsrb_ub_cooperative_L0_$i.txt
done
exit 0
