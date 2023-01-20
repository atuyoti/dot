import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os

filename = "10x10_test35"
inputfilepath = "./image/dot_without_ab_"+filename+"/"

outputfilepath = "./image/graph/intensity_graph/"+filename+"/"
if not os.path.isdir(outputfilepath):
	os.makedirs(outputfilepath)


import dot_parameter_test10x10_3
myClass = dot_parameter_test10x10_3.Dot()
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
num_detector = myClass.num_detector
pos_detector = myClass.pos_detector
num_light = myClass.num_light
pos_light = myClass.pos_light


def create_paramfile():
	print("unnko")
	file = outputfilepath+"param_"+filename+".txt"
	with open(file,'w') as f:
		f.write("stepnum_x:{0}\n".format(stepnum_x))
		f.write("stepnum_y:{0}\n".format(stepnum_y))
		f.write("dx:{0}\n".format(dx))
		f.write("dy:{0}\n".format(dy))
		f.write("dt:{0}\n".format(dt))
		f.write("stepnum_time:{0}\n".format(stepnum_time))

def main():
	#出力面の時間変化アレイを作成
	#y:出力面の座標
	#x:時間変化
	for index_light in range(num_light):
		phi= np.empty((stepnum_time,stepnum_x,stepnum_y))
		for i in range(stepnum_time):
			filename = "{0}-{1:03d}.npy".format(index_light,i)
			phi[i,:,:] = np.load(inputfilepath+filename)

		t = list(range(stepnum_time))
		t = np.array(t,dtype=float)

		fig = plt.figure()
		fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
		plt.ylim(0,0.0005)
		ax1.set_title("output intensity")
		for i in pos_detector:
			phi_output = phi[:,i[0],i[1]]
			ax1.plot(t,phi_output,label="{0},{1}".format(i[0],i[1]))
			ax1.legend()
		fig.savefig(outputfilepath+filename+".png")
		plt.clf()

		#3Dグラフ
		"""
		ax = fig.add_subplot(111, projection='3d')
		ax.set_title("Intensity of Output surface", size = 20)
		y = list(range(phi_output.shape[0]))
		X, Y = np.meshgrid(t, y)
		ax.plot_surface(X, Y, phi_output, cmap = "jet")
		ax.set_xlabel("time: dt=0.002 ps", size = 14)
		ax.set_ylabel("y_position", size = 14)
		plt.show()
		"""

if __name__ == '__main__':
	create_paramfile()
	main()