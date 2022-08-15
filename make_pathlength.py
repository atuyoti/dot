import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm
import itertools

import dot_parameter_test

outputfile = "./image/pathlength_test"
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
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time
H_map = np.zeros((stepnum_y,stepnum_x))
H_j=0
#phi_j=0


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
inputlight[int(center_y),0] = 10

#######################################
def diffuse(nt,y,x,t_d,h_xi_j,h_j_x):
	global H_j
	phi[:,0] = inputlight[:,nt]
	print("nt:%s : td:%s : %i" %(nt,t_d,inputlight[int(center_y),nt]))
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
		phi[:,-2] = phi_n[:,-3]*(2*D*A)/(2*D*A + dy)
		##x方向の境界条件
		phi[1,:] = phi_n[2,:]*(2*D*A)/(2*D*A + dx)
		phi[-2,:] = phi_n[-3,:]*(2*D*A)/(2*D*A + dx)

	#時刻t'の時のピクセルjの存在確率を求める
	if n==t_d and nt==t_d:
		if n%10 ==0:
			fig = plt.figure()
			fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
			ax1.set_title("nt="+str(nt))
			bar1=ax1.imshow(phi, cmap=cm.jet)
			fig.colorbar(bar1)
			fig.savefig(outputfile+"/png/{0:d}-{1:d}-{2:03d}-{3:03d}.png".format(i,j,t_d,nt))
			np.save(outputfile+"/{0:d}-{1:d}-{2:03d}-{3:03d}".format(i,j,t_d,nt),phi)

		print("####################")
		print("n=t_d")
		print("n={0},t_d={1}".format(n,t_d))
		print("####################")
		temp = np.copy(phi[y,x])

		#ピクセルjのみ残す
		phi[:,:]=0
		phi[y,x]=temp
		phi_j = phi[y,x]
		print("phi_j: "+str(phi_j))
		#入力光強度の比を存在確率とする
		if phi_j==0:
			h_xi_j = 0
		else:
			h_xi_j = phi_j/10
		print("h_xi_j: "+str(h_xi_j))


	#時刻t'でピクセルjに入射した光が時刻tにxで観測される確率
	if n == stepnum_time:
		print("n=max time")
		print("n={0},max_time={1}".foramt(n,stepnum_time))
		detector = myClass.pos_detector[0]
		h_j_x = phi[detector[0],detector[1]] / phi_j

		#時刻tでI(x,xi,t)を構成する光子が，時刻t'でピクセルjに存在する確率
		#h_j = h_xi_j * h_j_x
		#H_j = H_j + (h_j*dt)



	return h_xi_j,h_j_x


all_num = itertools.product(range(stepnum_y),range(stepnum_x))
for i, j in all_num:
	for t_d in range(stepnum_time):
		h_xi_j = -1
		h_j_x = -1
		for t in range(stepnum_time):
			h_xi_j,h_j_x = diffuse(t,i,j,t_d,h_xi_j,h_j_x)
			#時刻t'にピクセルjに光子が存在しない場合
			if h_xi_j==0:
				print("break")
				h_j = 0
				H_j = H_j + (h_j*dt)
				break
			#時刻t'にピクセルjに光子が存在し，時刻tにピクセルxに光子が存在した場合
			if h_xi_j>0 and h_j_x>0:
				h_j = h_xi_j * h_j_x
				H_j = H_j + (h_j*dt)
	H_map[i,j]=H_j

np.save(outputfile+"/H_map",H_map)
