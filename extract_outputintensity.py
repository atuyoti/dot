import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os

inputfilepath = "./image/dot_test2/"

dt =0.002
stepnum_time = 500
stepnum_y = 251
dt = 0.002 #ps

def main():
	phi_output = np.empty((int(stepnum_y),0))
	#出力面の時間変化アレイを作成
	#y:出力面の座標
	#x:時間変化
	for i in range(stepnum_time):
		filename = "{0:03d}.npy".format(i)
		phi = np.load(inputfilepath+filename)
		test = phi[:,-1]
		phi_output = np.append(phi_output,phi[:,-1].reshape(phi.shape[0],1),axis=1)
		print(phi_output.shape)

	t = list(range(stepnum_time))
	t = np.array(t,dtype=float)
	t = t*dt

	fig = plt.figure()
	fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
	ax1.set_title("output intensity")
	ax1.plot(t,phi_output[20,:],label="20")
	ax1.plot(t,phi_output[100,:],label="100")
	ax1.plot(t,phi_output[150,:],label="150")
	ax1.legend()
	fig.savefig("./image/intensity_graph/dot_test2_100.png")
	plt.clf()

	#3Dグラフ
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("Intensity of Output surface", size = 20)
	y = list(range(phi_output.shape[0]))
	X, Y = np.meshgrid(t, y)
	ax.plot_surface(X, Y, phi_output, cmap = "jet")
	ax.set_xlabel("time: dt=0.002 ps", size = 14)
	ax.set_ylabel("y_position", size = 14)
	plt.show()


if __name__ == '__main__':
	main()