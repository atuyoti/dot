import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os
import itertools

import dot_parameter_test

inputdir = "./image/graph/intensity_ratio/50x100_2"
filename = "/ratio.npy"
outputdir = "./image/graph/intensity_ratio/50x100_2"
myClass = dot_parameter_test.Dot()
stepnum_time = myClass.stepnum_time
stepnum_x = myClass.stepnum_x
stepnum_y = myClass.stepnum_y
dt = myClass.dt #ps
A = myClass.A
num_light = myClass.num_light
pos_light = myClass.pos_light
y = stepnum_y-2

def reconimage(ratio_array):

	rimage = np.zeros((num_light,stepnum_x,stepnum_y),dtype = np.float64)
	print(rimage.shape)
	for i in range(num_light):
		for j in range(1,stepnum_x-1):
			for k in range(1,stepnum_y-1):
				#print("y:{0},j:{1},k:{2},pos_light:{3}".format(y,j,k,pos_light[i,0]))
				#print("i:"+str(i))
				x_pos = (y/k)*(j-(((y-k)/y)*pos_light[i,0]))
				x_pos = round(x_pos)
				if x_pos<=0 or x_pos>=stepnum_x-1:
					rimage[i,j,k] = 0
				else:
					rimage[i,j,k] = (y/k)*ratio_array[x_pos,i]
			
	sumimage = np.sum(rimage,axis=0)
	print(sumimage.shape)
	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	ax1.set_title("reconimage")
	bar1 = ax1.imshow(sumimage, cmap=cm.jet,vmin = 0,vmax=10)
	fig.colorbar(bar1)
	fig.savefig(outputdir+"/reconimage-2.png")
	plt.close(fig)
	for i in range(num_light):
		fig = plt.figure()
		fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
		ax1.set_title("reconimage")
		bar1 = ax1.imshow(rimage[i,:,:], cmap=cm.jet,vmin = 0)
		fig.colorbar(bar1)
		fig.savefig(outputdir+"/{0}_recon.png".format(i))
		plt.close(fig)

ratio_array = np.load(inputdir+filename)
print(ratio_array.shape)
ratio_array[0,:]=0
ratio_array[stepnum_x-1,:]=0
reconimage(ratio_array)