import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from DataSet import *

log = open("ub_cifar10_resid_log.txt", "r")

l_bounds = []

f1 = log.readlines()
for x in f1:
	if x[:9] == 'Iteration':
		l_bounds.append(float(x[-11:-2]))
	print(x)

print(l_bounds)

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], hspace = 0.05)

ax = fig.add_subplot(gs[:, 0])
ax.plot(l_bounds)
ax.set_xlabel("Iteration number")
ax.set_ylabel("Upper bound (L2)")
# axes[0].set_xscale('log')
ax.set_aspect(50)

image = DataSet(data_set_name = 'cifar10', trainOrTest='test').get_input(314)
ax = fig.add_subplot(gs[0, 1])
ax.imshow(image)
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 1])
image = mpimg.imread('cifar10_pic/ub_cooperative/314_deer_modified_into_truck_with_confidence_0.40878743.png')
ax.imshow(image)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("test_figure_1.png")