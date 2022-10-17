import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm
import itertools

import dot_parameter_test20x20

outputfile = "./image/pathlength_20x20_3"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")

##################################
#条件設定
##################################
#ピコ秒での計測
myClass = dot_parameter_test20x20.Dot()
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
myu_a = myClass.myu_a_without
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time
accum_time = myClass.accum_time
accum_time_array = myClass.accum_time_array
H_j = np.zeros((accum_time_array.shape[0],stepnum_x,stepnum_y))
intensity = int(100)

num_detector = myClass.num_detector
pos_detector = myClass.pos_detector
##################################
#初期状態
##################################
phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
phi_n = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
#u[int(.5 / dy):int(1 / dy+1),int(.5 / dx):int(1 / dx + 1)] = 2
print(phi.shape)

#入力光源，あとで時間変化する形にする
inputlight = np.zeros((stepnum_x,stepnum_time))
num_light = myClass.num_light
pos_light = myClass.pos_light

#######################################


#xiに入射した光が時刻t'にjにいる確率
def calc_xi_to_j_probDens(phi,x,y):
	#in:強度分布phi,j(x,y)座標
	#out:存在確立h_xi_j,強度分布phi

	#ピクセルjのみ残す
	temp = np.copy(phi[x,y])
	phi[:,:]=0
	phi[x,y]=temp
	phi_j = phi[x,y]
	#print("phi_j: "+str(phi_j))

	#入力光強度の比を存在確率とする
	if phi_j==0:
		h_xi_j = 0
	else:
		h_xi_j = phi_j / intensity
	
	return h_xi_j,phi,phi_j

#時刻t'にjに入射した光が時刻tにxにいる確率
def calc_j_to_x_probDens(phi,phi_j,ndetec):
	detector = myClass.pos_detector[ndetec]
	h_j_x = phi[detector[0],detector[1]] / phi_j
	return h_j_x

#散乱の差分法
def diffuse(phi,nt):
	#in:強度分布phi
	#out:強度分布phi
	phi[:,0] = inputlight[:,nt]
	phi_n = phi.copy()

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

	return phi


def calc_H_j(x,y,nlight,phi,ac_time,ndetec):
	#in:(x,y)座標，初期化された強度分布phi，計測時間
	#out:あるピクセルの存在確立H_j(float)
	H_j_array = np.zeros((ac_time.shape[0]))
	inputlight[int(pos_light[nlight,0]),0] = intensity
	for t_d in tqdm(range(stepnum_time),leave=False,desc="t_d"):
		h_xi_j = -1
		h_j_x = -1
		H_j = 0
		index = 0
		ac_time_tmp = np.copy(ac_time)
		for t in range(stepnum_time):
			phi = diffuse(phi,t)
			#時刻t'の時のピクセルjの存在確率を求める
			if t==t_d:
				h_xi_j,phi,phi_j = calc_xi_to_j_probDens(phi,x,y)

			#時刻tの時のピクセルxの存在確立を求める
			if ac_time_tmp.size!=0 and t==ac_time_tmp[0] and t>=t_d:
				h_j_x = calc_j_to_x_probDens(phi,phi_j,ndetec)

			#時刻t'にピクセルjに光子が存在しない場合
			if h_xi_j==0:
				h_j = 0
				H_j = H_j + (h_j*dt)
				break
			elif h_xi_j>0 and h_j_x<=0:
				h_j =0
				H_j = H_j + (h_j*dt)
			#時刻t'にピクセルjに光子が存在し，時刻tにピクセルxに光子が存在した場合
			elif h_xi_j>0 and h_j_x>0:
				h_j = h_xi_j * h_j_x
				H_j = H_j + (h_j*dt)

			if ac_time_tmp.size!=0 and t == ac_time_tmp[0]:
				#tqdm.write("{0}-{1}".format(t,H_j))
				H_j_array[index] = H_j_array[index] + H_j
				index = index+1
				ac_time_tmp = np.delete(ac_time_tmp,0)
	return H_j_array


def test():
	for i in range(5):
		for j in range(i+1):
			print("{0}-{1}".format(i,j))


def main():
	for index_detec in tqdm(range(num_detector)):
		for index_light in tqdm(range(num_light),leave=False):
			all_num = itertools.product(range(1,stepnum_x-1),range(1,stepnum_y-1))
			for i, j in tqdm(all_num,desc="x,y",leave=False):
				#tqdm.write("{0}-{1}".format(i,j))
				phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
				H_j[:,i,j] = calc_H_j(i,j,index_light,phi,accum_time_array,index_detec)

			

		np.save(outputfile+"/H_map_{0:02d}-{1:02d}".format(index_light,index_detec),H_j)


if __name__ == '__main__':
	main()
	#test()