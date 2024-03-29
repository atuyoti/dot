import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm

import dot_parameter_test

outputfile = "./image/dot_with_ab_50x100_test2"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")

##################################
#条件設定
##################################
#ピコ秒での計測
myClass = dot_parameter_test.Dot()
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
#myu_a = myClass.myu_a_without
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time




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

##################################
#初期状態
##################################
phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
phi_n = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
#u[int(.5 / dy):int(1 / dy+1),int(.5 / dx):int(1 / dx + 1)] = 2
print(phi.shape)
 
#入力光源，あとで時間変化する形にする
inputlight = np.zeros((stepnum_x,stepnum_time))
center_x = stepnum_x/2
start_x = center_x-10
end_x = center_x+10
#inputlight[int(start_y):int(end_y),0:5] = 10
#inputlight[int(start_y):int(end_y),:] = myClass.pulse()
num_light = myClass.num_light
pos_light = myClass.pos_light
print(pos_light.shape)
#for i in range(num_light):
#	inputlight[int(pos_light[i,0]),:] = myClass.pulse()

#######################################
def diffuse(nt,nlight):
	#phi[:,0] = inputlight[:,nt]
	print("%s : %i" %(nt,inputlight[int(center_x),nt]))
	for n in range(nt + 1):
		phi_n = phi.copy()
		phi[:,0] = inputlight[:,n]
		phi[1:-1,1:-1] 	= c*dt*D*((phi_n[1:-1,2:] - 2 * phi_n[1:-1,1:-1] + phi_n[1:-1,0:-2]) / (dx**2)) \
						+ c*dt*D*((phi_n[2:,1:-1] -2 * phi_n[1:-1,1:-1] + phi_n[0:-2,1:-1]) / (dy**2)) \
						- ((c*dt*myu_a[1:-1,1:-1] - 1)*phi_n[1:-1,1:-1])
		#境界条件
		##y方向の境界条件
		#入力面での境界条件
		phi[:,1] = (1/(2*D*A+dy))*(2*D*A*phi_n[:,2] + (inputlight[:,nt] * (4*dy) / (1-rd)))

		#出力面での境界条件
		phi[:,-2] = phi_n[:,-3]*(2*D*A)/(2*D*A + dy)
		##x方向の境界条件
		phi[1,:] = phi_n[2,:]*(2*D*A)/(2*D*A + dx)
		phi[-2,:] = phi_n[-3,:]*(2*D*A)/(2*D*A + dx)

		if n ==0:
			fig = plt.figure()

			fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
			ax1.set_title("nt="+str(nt))
			bar1=ax1.imshow(phi, cmap=cm.jet,norm = colors.LogNorm(vmin = 0.00001,vmax =1))
			fig.colorbar(bar1)
			#plt.show()
			fig.savefig(outputfile+"/png/test-{0:d}-0.png".format(nt))
			plt.close(fig)


	fig = plt.figure()

	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	ax1.set_title("nt="+str(nt))
	bar1=ax1.imshow(phi, cmap=cm.jet,norm = colors.LogNorm(vmin = 0.00001,vmax =1))
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/png/{0:d}-{1:03d}.png".format(nlight,nt))
	np.save(outputfile+"/{0:d}-{1:03d}".format(nlight,nt),phi)
	print(phi[int(center_x),3])

for j in range(num_light):
	inputlight[int(pos_light[j,0]),:] = myClass.pulse()
	phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
	phi_n = np.zeros((stepnum_x,stepnum_y),dtype = np.float64) 
	#for i in range(stepnum_time):
	#	diffuse(i,j)
	diffuse(97,j)
	inputlight = np.zeros((stepnum_x,stepnum_time))