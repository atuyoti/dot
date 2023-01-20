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

#w = 0.001
w = 0.0005

#import dot_parameter_test10x10_3
#import dot_parameter_test10x20
#import dot_parameter_test20x20_2
#import dot_parameter_test32x32
#import dot_parameter_test32x32_2
import dot_parameter_test32x32_4

filename = "32x32_test_9absorb"
#filename2 = "10x10_test38"
#inputfile1 = "./image/dot_without_ab_10x10_test20/"
#inputfile2 = "./image/dot_with_ab_10x10_test20/"
#inputfile3 = "./image/pathlength_10x10_edited_myu_a_test/"
#inputfile3 = "./image/pathlength_10x10_test20/"

inputfile1 = "./image/dot_without_ab_"+filename+"/"
inputfile2 = "./image/dot_with_ab_"+filename+"/"
#inputfile3 = "./image/pathlength_10x10_edited_myu_a_test/"
inputfile3 = "./image/pathlength_"+filename+"/"



outputfile = "./image/reconimage"
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")

##################################
#条件設定
##################################
#ピコ秒での計測
#myClass = dot_parameter_test10x10_3.Dot()
#myClass = dot_parameter_test10x20.Dot()
#myClass = dot_parameter_test20x20_2.Dot()
#myClass = dot_parameter_test32x32_2.Dot()
myClass = dot_parameter_test32x32_4.Dot()
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
accum_time = myClass.accum_time 
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

#######################################
H_j = np.zeros((num_detector*num_light*accum_time_array.shape[0],(stepnum_x-2)*(stepnum_y-2)))

def create_paramfile():
	print("unnko")
	file = outputfile+"/param_"+filename+".txt"
	with open(file,'w') as f:
		f.write("stepnum_x:{0}\n".format(stepnum_x))
		f.write("stepnum_y:{0}\n".format(stepnum_y))
		f.write("dx:{0}\n".format(dx))
		f.write("dy:{0}\n".format(dy))
		f.write("dt:{0}\n".format(dt))
		f.write("stepnum_time:{0}\n".format(stepnum_time))

def load_and_calc_ratio(ac_time_array):

	B_data = np.zeros((ac_time_array.shape[0],stepnum_x,stepnum_y))
	T_data = np.zeros((ac_time_array.shape[0],stepnum_x,stepnum_y))
	B = np.zeros((num_detector*num_light,ac_time_array.shape[0]))
	T = np.zeros((num_detector*num_light,ac_time_array.shape[0]))
	#print(T_data.shape)

	for index_light in range(num_light):
		for index_detec in range(num_detector):
			for index,time in enumerate(ac_time_array):
				npyname = "{0}-{1:03d}.npy".format(index_light,time)
				filename_B = inputfile1+npyname #without
				filename_T = inputfile2+npyname #with
				B_data[index,:,:] = np.load(filename_B)
				T_data[index,:,:] = np.load(filename_T)
				#print((index_light*num_detector)+index_detec)
				B[(index_light*num_detector)+index_detec,index] = B_data[index,pos_detector[index_detec,0],pos_detector[index_detec,1]]
				T[(index_light*num_detector)+index_detec,index] = T_data[index,pos_detector[index_detec,0],pos_detector[index_detec,1]]


	B = np.where(np.isnan(B), 0 ,B)
	T = np.where(np.isnan(T), 0 ,T)
	y_i = B / T
	y_i = np.where(y_i!=0,np.log10(y_i),y_i)
	y_i = np.where(np.isnan(y_i), 0 ,y_i)
	y_i_reshape = np.reshape(y_i,(y_i.shape[0]*y_i.shape[1]))

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
	#対象領域のみにクロップ
	crop = array[1:-1,1:-1]
	return crop

def load_Hj_and_calc_lightpath():
	#光路長分布の配列
	L_j = np.zeros_like(H_j)

	for i in range(num_light):
		for j in range(num_detector):
			H_j_temp =  np.load(inputfile3+"H_map_{0:02d}-{1:02d}.npy".format(i,j))
			#H_j_temp_diff = np.diff(H_j_temp,axis=0)
			#H_j_temp[1:,:,:] = H_j_temp_diff
			print(H_j_temp.shape)
			for k,time in enumerate(accum_time_array):
				I_map = np.load(inputfile1+"{0}-{1:03d}.npy".format(i,time))
				I_r = I_map[pos_detector[j,0],pos_detector[j,1]] / (2*A)
				
				index = (i*num_detector*accum_time_array.shape[0]) + (j*accum_time_array.shape[0]) + k
				
				H_j_flat = convert2Dto1D(crop_array(H_j_temp[:,:]))
				H_j[index,:] = H_j_flat
				H_j_sum = np.sum(H_j,axis=1)
				L_j[index,:] = (H_j[index,:] / H_j_sum[index]) * c * time *dt
				#L_j[index,:] = (H_j[index,:] / I_r) * c
				#print(L_j.shape)

	return L_j





def cgm(A, b, x_init,num):
    x = x_init
    print(A.shape,x.shape)
    r0 = b - np.dot(A,x)
    p = r0
    k = 0
    for i in range(num):
        a = float( np.dot(r0.T,r0) / np.dot(np.dot(p.T, A),p) )
        x = x + p*a
        r1 = r0 - np.dot(A*a, p)
        if i% 1000 ==0:
       		print(i,np.linalg.norm(r1))
        b = float( np.dot(r1.T, r1) / np.dot(r0.T, r0) )
        p = r1 + b * p
        r0 = r1
    return x


def test2():
	y = load_and_calc_ratio(accum_time_array)
	L = load_Hj_and_calc_lightpath()
	myu_a_crop = crop_array(myu_a)
	myu_a_with_crop = crop_array(myu_a_with)
	x_ref = convert2Dto1D(myu_a_crop)
	x_0 = np.zeros_like(x_ref)
	L_T = L.T
	

	y_max = np.max(y)
	y_norm = y * 1 /y_max
	noise = random_noise(y_norm,mode="gaussian",var=0.00000001)
	#noise = random_noise(y_norm,mode="poisson")
	w0 = 0.0005
	y_noise = noise * y_max
	#y_noise = (y_norm + w0*noise) *y_max
	#y_noise = y

	print("y",y)
	print("y_norm",y_norm)
	print("noise",noise)
	print("y_noise",y_noise)

	#a = np.dot(L_T,L)/1.8
	#a = np.dot(L_T,L)/3
	a = np.dot(L_T,L)/2.4
	b = np.dot(L_T,y_noise)
	b_test = np.dot(L_T,y)


	ans1 = cgm(a,b_test,x_0,500)
	x_test = ans1+x_ref
	x_reshape1 = np.reshape(x_test,[stepnum_x-2,stepnum_y-2])

	ans2 = cgm(a,b_test,x_0,5000)
	x_test2 = ans2+x_ref
	x_reshape2 = np.reshape(x_test2,[stepnum_x-2,stepnum_y-2])

	x_denoise = ndimage.median_filter(x_reshape2,size=3)

	ans3 = cgm(a,b_test,x_0,20000)
	x_test3 = ans3+x_ref
	x_reshape3 = np.reshape(x_test3,[stepnum_x-2,stepnum_y-2])

	ans4 = cgm(a,b_test,x_0,30000)
	x_test4 = ans3+x_ref
	x_reshape4 = np.reshape(x_test4,[stepnum_x-2,stepnum_y-2])

	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	bar1=ax1.imshow(x_denoise, cmap=cm.Greys,vmin=0,vmax=0.02)
	#bar1=ax1.imshow(x_reshape1, cmap=cm.Greys,vmin=0)
	#bar1=ax1.imshow(x_reshape, cmap=cm.Greys)
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/test"+filename+"_denoise.png")
	plt.close(fig)
	np.save(outputfile+"/test"+filename+"_denoise",x_denoise)


	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	bar1=ax1.imshow(x_reshape1, cmap=cm.Greys_r,vmin=0,vmax=0.02)
	#bar1=ax1.imshow(x_reshape1, cmap=cm.Greys,vmin=0)
	#bar1=ax1.imshow(x_reshape, cmap=cm.Greys)
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/test"+filename+"_reconimage-500_r.png")
	plt.close(fig)

	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	bar1=ax1.imshow(x_reshape2, cmap=cm.Greys_r,vmin=0,vmax=0.02)
	#bar1=ax1.imshow(x_reshape2, cmap=cm.Greys,vmin=0)
	#bar1=ax1.imshow(x_reshape, cmap=cm.Greys)
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/test"+filename+"_reconimage-5000_r.png")
	plt.close(fig)
	np.save(outputfile+"/test"+filename+"_recon-5000",x_reshape2)

	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	bar1=ax1.imshow(x_reshape3, cmap=cm.Greys_r,vmin=0,vmax=0.02)
	#bar1=ax1.imshow(x_reshape3, cmap=cm.Greys,vmin=0)
	#bar1=ax1.imshow(x_reshape, cmap=cm.Greys)
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/test"+filename+"_reconimage-20000_r.png")
	plt.close(fig)

	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	bar1=ax1.imshow(x_reshape4, cmap=cm.Greys_r,vmin=0,vmax=0.02)
	#bar1=ax1.imshow(x_reshape3, cmap=cm.Greys,vmin=0)
	fig.colorbar(bar1)
	#plt.show()
	fig.savefig(outputfile+"/test"+filename+"_reconimage-30000_r.png")
	plt.close(fig)




if __name__ == '__main__':
	#main()
	create_paramfile()
	test2()
	#test()