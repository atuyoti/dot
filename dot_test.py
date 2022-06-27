import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
from tqdm import tqdm

##################################
#条件設定
##################################
stepnum_x = 51
stepnum_y = 51
length_x = 2
length_y = 2
nu = .05 #v
dx = length_x / (stepnum_x - 1)
dy = length_y / (stepnum_y - 1)
sigma = .25
dt = sigma * dx * dy /nu

x = np.linspace(0,length_x,stepnum_x)
y = np.linspace(0,length_y,stepnum_y)


##################################
#初期状態
##################################
u = np.ones((stepnum_y,stepnum_x))
un = np.ones((stepnum_y,stepnum_x))
#u[int(.5 / dy):int(1 / dy+1),int(.5 / dx):int(1 / dx + 1)] = 2
print(u.shape)
u[10:40,0:1] = 150

def diffuse(nt):
		for n in range(nt + 1):
			un = u.copy()
			u[1:-1,1:-1] = (un[1:-1,1:-1] + 
				nu * dt / dx**2 * 
				(un[1:-1,2:] -2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
				nu * dt / dy**2 *
				(un[2:,1:-1] -2 * un[1:-1,1:-1] + un[0:-2,1:-1]))

			u[0,:] = 1
			u[-1,:] = 1
			u[:,0] = 1
			u[:,-1] = 1

		fig = plt.figure()

		fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
		ax1.set_title("defoimage")
		bar1=ax1.imshow(u, cmap=cm.jet,vmin = 0, vmax = 3)
		fig.colorbar(bar1)
		#plt.show()
		fig.savefig("image1/{0:03d}.png".format(nt))


for i in range(50):
		diffuse(i)