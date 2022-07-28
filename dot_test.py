import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm

import dot_parameter

outputfile = "./image/dot_with_ab"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")

##################################
#条件設定
##################################
#ピコ秒での計測
myClass = dot_parameter.Dot()
stepnum_x = myClass.stepnum_x
stepnum_y = myClass.stepnum_y
length_x = myClass.length_x
length_y = myClass.length_y
dx = myClass.dx
dy = myClass.dy
dt = myClass.dt
g = myClass.g #非等方散乱パラメータ，あとで場所ごとに変わるように変更
myu_s = myClass.myu_s #散乱係数，あとで場所ごとに変わるように変更
c = myClass.c #mm/ps
D = myClass.D #光拡散係数
n_rel = myClass.n_rel
rd = myClass.rd
A = myClass.A
myu_a = myClass.myu_a_with
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time


######
fig = plt.figure()
fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
ax1.set_title("myu_a")
bar1=ax1.imshow(myu_a, cmap=cm.Greys)
fig.colorbar(bar1)
#plt.show()
fig.savefig(outputfile+"/png/myu_a.png")
np.save(outputfile+"/myu_a",myu_a)
plt.clf()
plt.close()
####

##################################
#初期状態
##################################
phi = np.zeros((stepnum_y,stepnum_x),dtype = np.float64)
phi_n = np.zeros((stepnum_y,stepnum_x),dtype = np.float64)
#u[int(.5 / dy):int(1 / dy+1),int(.5 / dx):int(1 / dx + 1)] = 2
print(phi.shape)

#入力光源，あとで時間変化する形にする
inputlight = np.zeros((stepnum_y,stepnum_time))
center_y = stepnum_y/2
start_y = center_y-10
end_y = center_y+10
#inputlight[int(start_y):int(end_y),0:5] = 10
inputlight[int(start_y):int(end_y),:] = myClass.pulse()

#######################################
def diffuse(nt):
	phi[:,0] = inputlight[:,nt]
	print("%s : %i" %(nt,inputlight[int(center_y),nt]))
	for n in range(nt + 1):
		phi_n = phi.copy()

		phi[1:-1,1:-1] 	= c*dt*D*((phi_n[1:-1,2:] - 2 * phi_n[1:-1,1:-1] + phi_n[1:-1,0:-2]) / (dx**2)) \
						+ c*dt*D*((phi_n[2:,1:-1] -2 * phi_n[1:-1,1:-1] + phi_n[0:-2,1:-1]) / (dy**2)) \
						- ((c*dt*myu_a[1:-1,1:-1] - 1)*phi_n[1:-1,1:-1])
		#境界条件
		##y方向の境界条件
		#入力面での境界条件
		phi[:,1] = (1/(2*D*A+dy))*(2*D*A*phi_n[:,1] + (inputlight[:,nt] * (4*dy) / (1-rd)))

		#出力面での境界条件
		phi[:,-1] = phi_n[:,-2]*(2*D*A)/(2*D*A + dy)
		##x方向の境界条件
		phi[1,:] = phi_n[2,:]*(2*D*A)/(2*D*A + dx)
		phi[-1,:] = phi_n[-2,:]*(2*D*A)/(2*D*A + dx)


	fig = plt.figure()

	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	ax1.set_title("nt="+str(nt))
	bar1=ax1.imshow(phi, cmap=cm.jet,norm = colors.LogNorm(vmin = 0.00001,vmax =1))
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/png/{0:03d}.png".format(nt))
	np.save(outputfile+"/{0:03d}".format(nt),phi)
	print(phi[int(center_y),3])


for i in range(stepnum_time):
		diffuse(i)