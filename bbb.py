import itertools
import numpy as np
from tqdm import tqdm
import dot_parameter_test2
myClass = dot_parameter_test2.Dot()
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
myu_a = myClass.myu_a_with
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time
accum_time = myClass.accum_time
accum_time_array = myClass.accum_time_array
H_j = np.zeros((accum_time_array.shape[0],stepnum_x,stepnum_y))
intensity = int(100)

num_detector = myClass.num_detector
pos_detector = myClass.pos_detector
num_light = myClass.num_light
pos_light = myClass.pos_light


def test1(ac_time):
	index = 0
	H_j_array = np.zeros((ac_time.shape[0]))
	for t in range(10):
		if t in ac_time:
			H_j_array[index] = 10
			index = index+1
	print(H_j_array.shape)
	return H_j_array

def test2(ac_time):
	index = 0
	for t_d in range(3):
		ac_time_tmp = np.copy(ac_time)
		for t in range(stepnum_time):
			print(t_d)
			print(ac_time_tmp)
			if t in ac_time_tmp and t==ac_time_tmp[index] and t>=t_d:
				print(t)
				h_j_x = 1
				H_j =0 
				ac_time_tmp = np.delete(ac_time_tmp,index)
			if t in ac_time_tmp:
				H_j_array[index] = H_j
				index = index+1

def test3():
	for index in range(3):
		print(index)
		all_num = itertools.product(range(3),range(3))
		for i, j in all_num:
			print("{0}-{1}-{2}".format(index,i,j))

#二点間の距離を計算
def calc_distance(x1,y1,x2,y2):
	distance = np.sqrt(((x1 - x2)*dx)**2 + ((y1 - y2)*dx)**2 )
	return distance

distance_xi_j = calc_distance(pos_light[0,0],pos_light[0,1],x,y)
