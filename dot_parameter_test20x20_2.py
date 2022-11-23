# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import os

class Dot:
	stepnum_x = 21
	stepnum_y = 21
	length_x = 0.4
	length_y = 0.4
	dx = length_x / (stepnum_x - 1)
	dy = length_y / (stepnum_y - 1)
	dt = 0.01 #ps 0.002
	g = 0 #非等方散乱パラメータ，あとで場所ごとに変わるように変更
	myu_s = 15 #散乱係数，あとで場所ごとに変わるように変更
	c = 0.225600000 #mm/ps
	D = 1 / (3*(1 - g)*myu_s) #光拡散係数
	n_rel = 1.33
	rd = -1.440/n_rel**2 + 0.710/n_rel + 0.668 + 0.0636*n_rel
	A = (1+rd) / (1-rd)
	myu_a_without = np.ones((stepnum_x,stepnum_y))*0.01 
	myu_tmp = np.copy(myu_a_without)
	myu_tmp[4:7,9:12] = 0.02
	myu_tmp[14:16,14:16] = 0.017
	myu_tmp[13:17,5:9] =0.02
	#myu_tmp[5:7,5:7] = 50
	myu_a_with = myu_tmp
	x = np.linspace(0,length_x,stepnum_x)
	y = np.linspace(0,length_y,stepnum_y)
	stepnum_time = 3001
	accum_time = 3000
	accum_time_array = np.arange(stepnum_time,step=accum_time)
	accum_time_array = np.delete(accum_time_array,0)
	num_detector = 23
	pos_detector = np.array([[1,3],[1,9],[1,15],[-2,6],[-2,12],[1,-2],[2,-2],[3,-2],[4,-2],[5,-2],[6,-2],[7,-2],[8,-2],[9,-2],[10,-2],[11,-2],[12,-2],[13,-2],[14,-2],[15,-2],[16,-2],[17,-2],[18,-2]])
	center_x = (stepnum_x-1)/2
	num_light = 17
	#pos_light = np.array([[center_y,0],[5,0],[15,0]])
	pos_light = np.array([[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],[11,0],[12,0],[13,0],[14,0],[15,0],[16,0],[17,0],[18,0]])

	def pulse(self,amp=10,t1=15,t2=5,dlen=stepnum_time,_dt=1):
		t = np.linspace(0,_dt*(dlen-1),dlen)
		y = amp * (np.exp(-((t-t1) ** 2)/(t2 ** 2)))
		fig = plt.figure()
		fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
		bar1=ax1.plot(t,y)

		fig.savefig("./image/test/pulse.png")
		plt.clf()
		plt.close()
		return y


myClass = Dot()

print(myClass.accum_time_array)
