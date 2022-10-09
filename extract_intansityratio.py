import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os


import dot_parameter_test2

inputfilepath1 = "./image/dot_with_ab_20x20_3light2detec/"
inputfilepath2 = "./image/dot_without_ab_20x20_3light2detec/"
outputfile = "./image/graph/intensity_ratio/20x20_3light2detec"

myClass = dot_parameter_test2.Dot()
stepnum_time = myClass.stepnum_time
stepnum_x = myClass.stepnum_x
dt = myClass.dt #ps
A = myClass.A
num_light = myClass.num_light

def main():
	if not os.path.isdir(outputfile):
		os.makedirs(outputfile)

	p_ratio_array = np.empty((int(stepnum_x),0))
	for j in range(num_light):
		phi_output_with = np.empty((int(stepnum_x),0))
		phi_output_without = np.empty((int(stepnum_x),0))

		for i in range(stepnum_time):
			filename = "{0:d}-{1:03d}.npy".format(j,i)
			#出力強度分布に変換
			phi_with_ab = np.load(inputfilepath1+filename)/(2*A)
			phi_without_ab = np.load(inputfilepath2+filename)/(2*A)

			#出力面の出力強度分布を時間方向に並べる
			phi_output_with = np.append(phi_output_with,phi_with_ab[:,-2].reshape(phi_with_ab.shape[0],1),axis=1)
			phi_output_without = np.append(phi_output_without,phi_without_ab[:,-2].reshape(phi_without_ab.shape[0],1),axis=1)

	 		
			"""
			phi_ratio = phi_with_ab / phi_without_ab
			phi_log = np.zeros_like(phi_ratio)
			
			for j in range(phi_ratio.shape[0]):
				for l in range(phi_ratio.shape[1]):
					if phi_ratio[j,l]==0:
						pass
					else:
						phi_log[j,l] = - math.log(phi_ratio[j,l])

			np.save(outputfile+"/{0:d}-{1:03d}".format(j,i),phi_log)
			fig = plt.figure()
			fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
			ax1.set_title("intensity ratio:"+str(i))
			bar1 = ax1.imshow(phi_log, cmap=cm.jet,vmin = 0)
			fig.colorbar(bar1)
			fig.savefig(outputfile+"/{0:03d}.png".format(i))
			plt.close(fig)
			"""
		print(phi_output_with.shape)
		#強度分布の時間積分
		p_m = np.sum(phi_output_with,axis=1)*dt
		p_r = np.sum(phi_output_without,axis=1)*dt
		print(p_r.shape)
		#出力面の強度比分布
		p_ratio = p_m / p_r
		p_ratio = np.where(p_ratio!=0,-np.log(p_ratio),p_ratio)
		print(p_ratio)
		print(p_ratio.shape)
		#出力面の強度比分布の入力光順に並べる
		p_ratio_array = np.append(p_ratio_array,p_ratio.reshape(p_ratio.shape[0],1),axis=1)
		print(p_ratio_array.shape)
		np.save(outputfile+"/ratio",p_ratio_array)


if __name__ == '__main__':
	main()