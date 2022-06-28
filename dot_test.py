import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
from tqdm import tqdm

outputfile = "./image/dot_test"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)

##################################
#条件設定
##################################
#ナノ秒での計測
stepnum_x = 501
stepnum_y = 501
length_x = 100
length_y = 100
nu = .05 #v
dx = length_x / (stepnum_x - 1)
dy = length_y / (stepnum_y - 1)
sigma = .25
#dt = sigma * dx * dy /nu
dt = 1
g = 0 #非等方散乱パラメータ，あとで場所ごとに変わるように変更
myu_s = 1 #散乱係数，あとで場所ごとに変わるように変更
c = 0.299792458 #mm/ps
D = 1 / (3*(1 - g)*myu_s) #光拡散係数
myu_a = np.ones((stepnum_y,stepnum_x)) #吸収率，あとでnumpy arrayから持ってくるように変更
x = np.linspace(0,length_x,stepnum_x)
y = np.linspace(0,length_y,stepnum_y)
stepnum_time = 50

##################################
#初期状態
##################################
#u = np.ones((stepnum_y,stepnum_x))
#un = np.ones((stepnum_y,stepnum_x))
phi = np.ones((stepnum_y,stepnum_x))
phi_n = np.ones((stepnum_y,stepnum_x))
#u[int(.5 / dy):int(1 / dy+1),int(.5 / dx):int(1 / dx + 1)] = 2
print(phi.shape)

#入力光源，あとで時間変化する形にする
inputlight = np.zeros(stepnum_time)
inputlight[0:5] = 100
center_y = stepnum_y/2
start_y = center_y-10
end_y = center_y+10
#phi[int(start_y):int(end_y),0:2] = 150

#######################################
def diffuse(nt):
	phi[int(start_y):int(end_y),0:2] = inputlight[nt]
	print("%s : %i" %(nt,inputlight[nt]))
	for n in range(nt + 1):
		phi_n = phi.copy()
		"""
		u[1:-1,1:-1] = (un[1:-1,1:-1] + 
				nu * dt / dx**2 * 
				(un[1:-1,2:] -2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
				nu * dt / dy**2 *
				(un[2:,1:-1] -2 * un[1:-1,1:-1] + un[0:-2,1:-1]))
		"""
		phi[1:-1,1:-1] 	= (c*dt*D*(phi_n[1:-1,2:] -2 * phi_n[1:-1,1:-1] + phi_n[1:-1,0:-2]) /dx**2
						+ c*dt*D*(phi_n[2:,1:-1] -2 * phi_n[1:-1,1:-1] + phi_n[0:-2,1:-1]) /dy**2
						- (c*dt*myu_a[1:-1,1:-1] -1)*phi_n[1:-1,1:-1])

		#境界条件
		"""
		phi[0,:] = 1
		phi[-1,:] = 1
		phi[:,0] = 1
		phi[:,-1] = 1
		"""
	fig = plt.figure()

	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	ax1.set_title("defoimage")
	bar1=ax1.imshow(phi, cmap=cm.jet,vmin = 0, vmax = 3)
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/{0:03d}.png".format(nt))

	print(phi[int(center_y),3])


for i in range(stepnum_time):
		diffuse(i)