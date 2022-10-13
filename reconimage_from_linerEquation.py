import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm
import itertools
import math

import dot_parameter_test2

inputfile1 = "./image/dot_without_ab_20x20_3light2detec/"
inputfile2 = "./image/dot_with_ab_20x20_3light2detec/"
inputfile3 = "./image/pathlength_test3/"

outputfile = "./image/pathlength_test3"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")

##################################
#条件設定
##################################
#ピコ秒での計測
myClass = dot_parameter_test2.Dot()
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

intensity = int(100)

num_detector = myClass.num_detector
pos_detector = myClass.pos_detector
##################################
#初期状態
##################################
phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
phi_n = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
#u[int(.5 / dy):int(1 / dy+1),int(.5 / dx):int(1 / dx + 1)] = 2

#入力光源，あとで時間変化する形にする
inputlight = np.zeros((stepnum_x,stepnum_time))
num_light = myClass.num_light
pos_light = myClass.pos_light

#######################################
H_j = np.zeros((num_detector*num_light*accum_time_array.shape[0],(stepnum_x-2)*(stepnum_y-2)))

#測定データのログ比y_iを出す
def load_and_calc_ratio(ac_time_array):

	B_data = np.zeros((ac_time_array.shape[0],stepnum_x,stepnum_y))
	T_data = np.zeros((ac_time_array.shape[0],stepnum_x,stepnum_y))
	B = np.zeros((num_detector*num_light,ac_time_array.shape[0]))
	T = np.zeros((num_detector*num_light,ac_time_array.shape[0]))
	print(T_data.shape)

	for index_light in range(num_light):
		for index,time in enumerate(ac_time_array):
			npyname = "{0}-{1:03d}.npy".format(index_light,time)
			filename_B = inputfile1+npyname
			filename_T = inputfile2+npyname
			B_data[index,:,:] = np.load(filename_B)
			T_data[index,:,:] = np.load(filename_T)

			B[index_light,index] = B_data[index,pos_detector[0,0],pos_detector[0,1]]
			B[index_light+1,index]=B_data[index,pos_detector[1,0],pos_detector[1,1]]
			T[index_light,index] = T_data[index,pos_detector[0,0],pos_detector[0,1]]
			T[index_light+1,index]=T_data[index,pos_detector[1,0],pos_detector[1,1]]

	y_i = B / T
	y_i = np.where(y_i!=0,-np.log(y_i),y_i)
	y_i_reshape = np.reshape(y_i,(y_i.shape[0]*y_i.shape[1]))
	print(y_i.shape)
	print(y_i_reshape.shape)
	"""
	light num - detector num - timestep
	0 - 0 - 10
	0 - 0 - 20
	:
	:
	0 - 0 - max
	0 - 1 - 10
	:
	1 - 0 - 10
	"""
	return y_i_reshape


def convert2Dto1D(test_array):
	test_ = np.reshape(test_array,(test_array.shape[0]*test_array.shape[1]))
	"""
	[[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]]
     to
	[ 0  1  2  3  4  5  6  7  8  9 10 11]
	"""
	return test_

def crop_array(array):
	crop = array[1:-1,1:-1]
	return crop

def load_Hj_and_calc_lightpath():
	#光路長分布の配列
	L_j = np.zeros_like(H_j)

	for i in range(num_light):
		for j in range(num_detector):
			H_j_temp =  np.load(inputfile3+"H_map_{0:02d}-{1:02d}.npy".format(i,j))

			for k,time in enumerate(accum_time_array):
				index = (i*num_detector*accum_time_array.shape[0]) + (j*accum_time_array.shape[0]) + k
				H_j_flat = convert2Dto1D(crop_array(H_j_temp[k,:,:]))
				H_j[index,:] = H_j_flat
				H_j_sum = np.sum(H_j,axis=1)
				L_j[index,:] = H_j[index,:] / H_j_sum[index] * c * time
	
	return L_j




def test():

	load_Hj_and_calc_lightpath()
	return

def main():
	y = load_and_calc_ratio(accum_time_array)
	myu_a_crop = crop_array(myu_a)
	x_ref = convert2Dto1D(myu_a_crop)
	x = np.zeros_like(x_ref)
	L = load_Hj_and_calc_lightpath()
	print("y:"+str(y.shape))
	print("x_ref:"+str(x_ref.shape))
	print("L:"+str(L.shape))


#for index_light in tqdm(range(num_light),leave=False):


if __name__ == '__main__':
	main()
	#test()