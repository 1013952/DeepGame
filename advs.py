import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from DataSet import *

indices = {
		'cifar10': [510, 509, 508, 507, 506],
		'gtsrb': [510, 505, 606, 603, 604]

}

fig, axes = plt.subplots(6, 3, figsize=(8, 12))
for row in axes:
	for ax in row:
		ax.set_xticks([])
		ax.set_yticks([])

axes[0, 0].set_title("Original")
axes[0, 1].set_title("Difference")
axes[0, 2].set_title("Adversary")

for i, dataset in enumerate(['cifar10', 'gtsrb']):
	dset = DataSet(data_set_name = dataset, trainOrTest = 'test')
	for j, idx in enumerate(indices[dataset][:3]):
		image = dset.get_input(idx)

		diffimg = mpimg.imread('%s_pic/ub_cooperative/%s_diff.png' %
						(dataset, idx))

		diffimg = diffimg[:, :, :3]

		print("i =  %d" % i)
		print("j = %d" % j)
		axes[3*i + j][0].imshow(image)
		axes[3*i + j][1].imshow(diffimg)
		axes[3*i + j][2].imshow(image + diffimg)
		print("Axes %d with dataset %s" % (5*i +j, dataset))

plt.savefig('figures/adversaries.png')


