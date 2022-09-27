import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os


import dot_parameter_test

inputfilepath1 = "./image/dot_with_ab_20x20_4/"
inputfilepath2 = "./image/dot_without_ab_20x20_4/"
outputfile = "./image/graph/intensity_ratio/20x20_4"

myClass = dot_parameter_test.Dot()
stepnum_time = myClass.stepnum_time
stepnum_y = myClass.stepnum_y
stepnum_x = myClass.stepnum_x
dt = myClass.dt #ps
A = myClass.A
num_light = myClass.num_light

phi = np.zeros((stepnum_y,stepnum_x),dtype = np.float64)
phi[1,-2] =100
phi[5,stepnum_y-1] = 100

fig = plt.figure()
fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
ax1.set_title("intensity ratio:")
bar1 = ax1.imshow(phi, cmap=cm.jet,vmin = 0)
fig.colorbar(bar1)
fig.savefig(outputfile+"/test.png")
plt.close(fig)