import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm
import itertools

file1 = "./image/pathlength_10x10_test29_0.01"
file2 = "./image/pathlength_10x10_test30"
outputfile = "./image/pathlength_10x10_test15"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)


import dot_parameter_test10x10_myu_a_edited

myClass = dot_parameter_test10x10_myu_a_edited.Dot()
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
intensity = int(1)

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
#時刻t'にjに入射した光が時刻tにxにいる確率
def calc_j_to_x_probDens(phi,ndetec,h_xi_j):
	detector = myClass.pos_detector[ndetec]
	h_j_x = phi[detector[0],detector[1]]
	return h_j_x

def calc_H_j_2(x,y,nlight,ac_time,ndetec):
	#in:(x,y)座標，初期化された強度分布phi，計測時間
	#out:あるピクセルの存在確立H_j(float)
	H_j_array = np.zeros((ac_time.shape[0]))
	for t_d in tqdm(range(stepnum_time),leave=False,desc="t_d"):
		h_xi_j = 0
		h_j_x = 0
		H_j = 0
		index = 0
		ac_time_tmp = np.copy(ac_time)
		phi_xi_j = np.load(file1+"/xi_j"+"/intensity_{0:02d}-{1:03d}.npy".format(nlight,t_d))
		h_xi_j = phi_xi_j[x,y]
		for index,time in enumerate(ac_time_tmp):
			t = time - t_d
			phi_j_x = np.load(file1+"/j_x"+"/intensity_{0:02d}-{1:02d}-{2:02d}.npy".format(x,y,t))
			h_j_x = calc_j_to_x_probDens(phi_j_x,ndetec,h_xi_j)

			#時刻t'にピクセルjに光子が存在しない場合
			if h_xi_j==0:
				h_j = 0
				H_j = H_j + (h_j*dt)
				break

			#時刻t'にピクセルjに光子が存在する場合
			elif h_xi_j>0:
				h_j = h_xi_j * h_j_x
				H_j = H_j + (h_j*dt)

			H_j_array[index] = H_j_array[index] + H_j

	return H_j_array



def main():
	"""
	for index_detec in tqdm(range(num_detector)):
		for index_light in tqdm(range(num_light),leave=False):
			all_num = itertools.product(range(1,stepnum_x-1),range(1,stepnum_y-1))
			for i, j in tqdm(all_num,desc="x,y",leave=False):
				H_j[:,i,j] = calc_H_j_2(i,j,index_light,accum_time_array,index_detec)

			np.save(outputfile+"/H_map_{0:02d}-{1:02d}".format(index_light,index_detec),H_j)
	"""
	for i in range(num_light):
		for j in range(num_detector):
			path1 = np.load(file1+"/H_map_{0:02d}-{1:02d}.npy".format(i,j))
			path2 = np.load(file2+"/H_map_{0:02d}-{1:02d}.npy".format(i,j))
			diff  = path2 - path1

			fig = plt.figure()
			fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
			bar1=ax1.imshow(diff[:,:], cmap=cm.jet)
							#bar1=ax1.imshow(phi, cmap=cm.jet)
			fig.colorbar(bar1)
							#plt.show()
			fig.savefig("./image/test/test_{0:02d}-{1:02d}.png".format(i,j))
			plt.clf()
			plt.close(fig)

			fig = plt.figure()
			fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 4.5),sharex=True, sharey=True)
			bar1=ax1.imshow(path1[:,:], cmap=cm.jet)
							#bar1=ax1.imshow(phi, cmap=cm.jet)
			bar2=ax2.imshow(path2[:,:], cmap=cm.jet)
							#bar1=ax1.imshow(phi, cmap=cm.jet)
			fig.colorbar(bar1,ax=ax1)
			fig.colorbar(bar2,ax=ax2)
							#plt.show()
			fig.savefig("./image/test/H_map_{0:02d}-{1:02d}.png".format(i,j))
			plt.clf()
			plt.close(fig)
	
def test():
	array = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
			array[i,j] = i*5 + j
	print(array)
	print(array[1:-1,1:-1])
	print(array[1:-1,2:])
if __name__ == '__main__':
	#test()
	main()