import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os
import math

import dot_parameter

inputfilepath1 = "./image/dot_with_ab/"
inputfilepath2 = "./image/dot_without_ab/"
outputfile = "./image/graph/intensity_ratio"

myClass = dot_parameter.Dot()
stepnum_time = myClass.stepnum_time
stepnum_y = myClass.stepnum_y
dt = myClass.dt #ps

def main():
	if not os.path.isdir(outputfile):
		os.makedirs(outputfile)


	for i in range(stepnum_time):
		filename = "{0:03d}.npy".format(i)
		phi_with_ab = np.load(inputfilepath1+filename)
		phi_without_ab = np.load(inputfilepath2+filename)
		phi_ratio = phi_with_ab / phi_without_ab
		phi_log = np.zeros_like(phi_ratio)
		for j in range(phi_ratio.shape[0]):
			for l in range(phi_ratio.shape[1]):
				if phi_ratio[j,l]==0:
					pass
				else:
					phi_log[j,l] = - math.log(phi_ratio[j,l])

		np.save(outputfile+"/{0:03d}".format(i),phi_log)
		fig = plt.figure()
		fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
		ax1.set_title("intensity ratio:"+str(i))
		bar1 = ax1.imshow(phi_log, cmap=cm.jet,vmin = 0,vmax =30)
		fig.colorbar(bar1)
		fig.savefig(outputfile+"/{0:03d}.png".format(i))
		plt.close(fig)


if __name__ == '__main__':
	main()