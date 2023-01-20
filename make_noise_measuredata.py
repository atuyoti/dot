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
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage
from skimage.util import random_noise
from PIL import Image
filename = "32x32_test4"
inputfile2 = "./image/dot_with_ab_"+filename+"/"

import dot_parameter_test32x32_4


##################################
#条件設定
##################################
#ピコ秒での計測
#myClass = dot_parameter_test10x10_3.Dot()
#myClass = dot_parameter_test10x20.Dot()
#myClass = dot_parameter_test20x20_2.Dot()
myClass = dot_parameter_test32x32_4.Dot()
#myClass = dot_parameter_test32x32_2.Dot()
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
myu_a_with = myClass.myu_a_with
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time 
accum_time = 10 
#accum_time_array = myClass.accum_time_array
accum_time_array = np.arange(stepnum_time,step=accum_time)
accum_time_array = np.delete(accum_time_array,0)

intensity = int(1)

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

print(pos_detector)



def crop_array(array):
	#対象領域のみにクロップ
	crop = array[1:-1,1:-1]
	return crop

def load_detec(ac_time_array):

	T_data = np.zeros((ac_time_array.shape[0],stepnum_x,stepnum_y))
	T = np.zeros(ac_time_array.shape[0])
	#print(T_data.shape)

	for index,time in enumerate(ac_time_array):
		npyname = "16-{0:03d}.npy".format(time)
		filename_T = inputfile2+npyname #with
		T_data[index,:,:] = np.load(filename_T)
		#print((index_light*num_detector)+index_detec)
		
	T = T_data[:,17,-2]


	return T


T = load_detec(accum_time_array)
T_max = np.max(T)
T_norm =  T * 1 /T_max
noise = random_noise(T_norm,mode="poisson")
noise_norm = noise *T_max

fig = plt.figure()
fig, ax1= plt.subplots(1, 1, figsize=(16, 8),sharex=True, sharey=True)
ax1.plot(accum_time_array,T)

#ax1.legend()
fig.savefig("test3.png")
plt.close(fig)

fig = plt.figure()
fig, ax1= plt.subplots(1, 1, figsize=(16, 8),sharex=True, sharey=True)
ax1.plot(accum_time_array,noise)

#ax1.legend()
fig.savefig("test3_noise.png")
plt.close(fig)

myu_a_crop = crop_array(myu_a_with) 
fig = plt.figure()
fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
bar1=ax1.imshow(myu_a_crop, cmap=cm.Greys_r, vmin=0,vmax=0.02)
#bar1=ax1.imshow(phi, cmap=cm.jet)
fig.colorbar(bar1)
#plt.show()
fig.savefig("myu_a_9absorb.png")
plt.clf()
plt.close(fig)