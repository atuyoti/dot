import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm
import itertools

#import dot_parameter_test10x10_myu_a_edited
#import dot_parameter_test10x10_2
import dot_parameter_test10x20
#import dot_parameter_test20x20_2

#filename = "pathlength_20x20_test7"
filename = "pathlength_10x20_test3"
#filename = "pathlength_10x10_test21"
outputfile = "./image/"+filename
if not os.path.isdir(outputfile):
	os.makedirs(outputfile)
	os.makedirs(outputfile+"/png")
	os.makedirs(outputfile+"/xi_j")
	os.makedirs(outputfile+"/j_x")

##################################
#条件設定
##################################
#ピコ秒での計測
#myClass = dot_parameter_test10x10_myu_a_edited.Dot()
#myClass = dot_parameter_test20x20_2.Dot()
myClass = dot_parameter_test10x20.Dot()
#myClass = dot_parameter_test10x10_2.Dot()
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

#散乱の差分法
def diffuse_input_from_surface(phi,nt,inputlight):
	#out:強度分布phi
	#phi[:,0] = inputlight[:,nt]
	phi_n = phi.copy()

	phi[1:-1,1:-1] 	= c*dt*D*((phi_n[1:-1,2:] - 2 * phi_n[1:-1,1:-1] + phi_n[1:-1,0:-2]) / (dx**2)) \
					+ c*dt*D*((phi_n[2:,1:-1] -2 * phi_n[1:-1,1:-1] + phi_n[0:-2,1:-1]) / (dy**2)) \
					- ((c*dt*myu_a[1:-1,1:-1] - 1)*phi_n[1:-1,1:-1])


	#境界条件
	##y方向の境界条件
	#入力面での境界条件
	#phi[:,0] = (1/(2*D*A+dy))*(2*D*A*phi_n[:,1] + (inputlight[:,nt] * (4*dy) / (1-rd)))
	phi[:,1] = (1/(2*D*A+dy))*(2*D*A*phi_n[:,2] + (inputlight[:,nt] * (4*dy) / (1-rd)))

	#出力面での境界条件
	#phi[:,-1] = phi_n[:,-2]*(2*D*A)/(2*D*A + dy)
	phi[:,-2] = phi_n[:,-3]*(2*D*A)/(2*D*A + dy)
	
	##x方向の境界条件
	#phi[0,:] = phi_n[1,:]*(2*D*A)/(2*D*A + dx)
	#phi[-1,:] = phi_n[-2,:]*(2*D*A)/(2*D*A + dx)
	phi[1,:] = phi_n[2,:]*(2*D*A)/(2*D*A + dx)
	phi[-2,:] = phi_n[-3,:]*(2*D*A)/(2*D*A + dx)



	return phi

def diffuse_input_inside(phi,nt):
	phi_n = phi.copy()

	phi[1:-1,1:-1] 	= c*dt*D*((phi_n[1:-1,2:] - 2 * phi_n[1:-1,1:-1] + phi_n[1:-1,0:-2]) / (dx**2)) \
					+ c*dt*D*((phi_n[2:,1:-1] -2 * phi_n[1:-1,1:-1] + phi_n[0:-2,1:-1]) / (dy**2)) \
					- ((c*dt*myu_a[1:-1,1:-1] - 1)*phi_n[1:-1,1:-1])


	#境界条件
	##y方向の境界条件
	#出力面での境界条件
	#phi[:,0] = phi_n[:,1]*(2*D*A)/(2*D*A + dy)
	#phi[:,-1] = phi_n[:,-2]*(2*D*A)/(2*D*A + dy)
	phi[:,1] = phi_n[:,2]*(2*D*A)/(2*D*A + dy)
	#phi[:,1] = (1/(2*D*A+dy))*(2*D*A*phi_n[:,2] + (inputlight[:,nt] * (4*dy) / (1-rd)))
	phi[:,-2] = phi_n[:,-3]*(2*D*A)/(2*D*A + dy)
	##x方向の境界条件
	phi[1,:] = phi_n[2,:]*(2*D*A)/(2*D*A + dx)
	phi[-2,:] = phi_n[-3,:]*(2*D*A)/(2*D*A + dx)
	#phi[0,:] = phi_n[1,:]*(2*D*A)/(2*D*A + dx)
	#phi[-1,:] = phi_n[-2,:]*(2*D*A)/(2*D*A + dx)



	return phi




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
		phi_xi_j = np.load(outputfile+"/xi_j"+"/intensity_{0:02d}-{1:03d}.npy".format(nlight,t_d))
		h_xi_j = phi_xi_j[x,y]
		for index,time in enumerate(ac_time_tmp):
			if t_d <= time:
				t = time - t_d
				phi_j_x = np.load(outputfile+"/j_x"+"/intensity_{0:02d}-{1:02d}-{2:02d}.npy".format(x,y,t))
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

	for index_light in tqdm(range(num_light),leave=False):
		inputlight = np.zeros((stepnum_x,stepnum_time))
		inputlight[int(pos_light[index_light,0]),0] = intensity
		phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
		for time in range(stepnum_time):
			phi[:,0] = inputlight[:,time]
			"""
			if time % 10 == 0:
				fig = plt.figure()
				fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
				ax1.set_title("nt="+str(time))
				bar1=ax1.imshow(phi, cmap=cm.jet,norm = colors.LogNorm(vmin = 0.00001,vmax = 0.1))
				#bar1=ax1.imshow(phi, cmap=cm.jet)
				fig.colorbar(bar1)
				#plt.show()
				fig.savefig(outputfile+"/png/{0:02d}-{1:03d}.png".format(index_light,time))
				plt.clf()
				plt.close(fig)
			"""
			np.save(outputfile+"/xi_j"+"/intensity_{0:02d}-{1:03d}".format(index_light,time),phi)
			phi = diffuse_input_from_surface(phi,time,inputlight)

	all_num = itertools.product(range(1,stepnum_x-1),range(1,stepnum_y-1))
	for i, j in tqdm(all_num,desc="x,y",leave=False):
		phi = np.zeros((stepnum_x,stepnum_y),dtype = np.float64)
		phi[i,j] = intensity
		for time in range(stepnum_time):
			
			if time % 50 == 0:
				fig = plt.figure()
				fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
				ax1.set_title("nt="+str(time))
				bar1=ax1.imshow(phi, cmap=cm.jet,norm = colors.LogNorm(vmin = 0.00001,vmax =0.1))
				fig.colorbar(bar1)
				#plt.show()
				fig.savefig(outputfile+"/png/j_x-{0:02d}-{1:02d}-{2:03d}.png".format(i,j,time))
				plt.clf()
				plt.close(fig)
			
			np.save(outputfile+"/j_x"+"/intensity_{0:02d}-{1:02d}-{2:02d}".format(i,j,time),phi)
			phi = diffuse_input_inside(phi,time)
			#np.save(outputfile+"/j_x"+"/intensity_nlight-{0:02d}-{1:02d}-{2:02d}-{3:03d}".format(index_light,i,j,time),phi)
			


			

	
	
	for index_detec in tqdm(range(num_detector)):
		for index_light in tqdm(range(num_light),leave=False):
			all_num = itertools.product(range(1,stepnum_x-1),range(1,stepnum_y-1))
			for i, j in tqdm(all_num,desc="x,y",leave=False):
				H_j[:,i,j] = calc_H_j_2(i,j,index_light,accum_time_array,index_detec)

			np.save(outputfile+"/H_map_{0:02d}-{1:02d}".format(index_light,index_detec),H_j)
	


if __name__ == '__main__':
	main()
	#test()