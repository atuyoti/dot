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

import  dot_parameter_test10x10_myu_a_edited

filename = "pathlength_10x10_test5"
outputfile = "./image/"+filename

if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")

##################################
#条件設定
##################################
#ピコ秒での計測
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

#二点間の距離を計算
def calc_distance(x1,y1,x2,y2):
	distance = np.sqrt(((x1 - x2)*dx)**2 + ((y1 - y2)*dx)**2 )
	return distance

#存在確立の計算
def calc_exisProb(distance,B):
	#Microscopic beer-lambert法則
	myu_t = myu_a[5,5]
	intensity =  B*math.exp(- myu_t *distance)
	prob = intensity / B
	return prob, intensity

def calc_H_j(x,y,nlight,ac_time,ndetec):
	#in:(x,y)座標，初期化された強度分布phi，計測時間
	#out:あるピクセルの存在確立H_j(float)
	H_j_array = np.zeros((ac_time.shape[0]))
	H_j = 0
	distance_xi_j = calc_distance(pos_light[nlight,0],pos_light[nlight,1],x,y)
	distance_j_x = calc_distance(x,y,pos_detector[ndetec,0],pos_detector[ndetec,1])
	ac_time_tmp = np.copy(ac_time)
	for index,t in enumerate(tqdm(ac_time_tmp,leave=False,desc="t")):
		H_j = 0
		for t_d in tqdm(range(stepnum_time),leave=False,desc="t_d"):
			if t_d>t:
				break
			#光が進んだ距離を計算
			optical_length_xi_j = c*t_d*dt
			optical_length_j_x = c*(t-t_d)*dt

			#光が進んだ距離が二点間の距離より長いとき
			if optical_length_xi_j>distance_xi_j:
				h_xi_j,phi_j = calc_exisProb(optical_length_xi_j,intensity)
			else:
				h_xi_j = 0

			if optical_length_j_x>distance_j_x and h_xi_j>0:
				h_j_x, phi_x = calc_exisProb(optical_length_j_x,phi_j)
			else:
				h_j_x = 0

			h_j = h_xi_j * h_j_x
			H_j = H_j + (h_j*dt)
		H_j_array[index] = H_j

	return H_j_array



for index_detec in tqdm(range(num_detector)):
	for index_light in tqdm(range(num_light),leave=False):
		all_num = itertools.product(range(1,stepnum_x-1),range(1,stepnum_y-1))
		for i, j in tqdm(all_num,desc="x,y",leave=False):
			H_j[:,i,j] = calc_H_j(i,j,index_light,accum_time_array,index_detec)
			

		np.save(outputfile+"/H_map_{0:02d}-{1:02d}".format(index_light,index_detec),H_j)


