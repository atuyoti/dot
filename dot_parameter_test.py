# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import os

class Dot:
	stepnum_x = 51
	stepnum_y = 51
	length_x = 1
	length_y = 1
	dx = length_x / (stepnum_x - 1)
	dy = length_y / (stepnum_y - 1)
	dt = 0.002
	g = 0 #非等方散乱パラメータ，あとで場所ごとに変わるように変更
	myu_s = 10 #散乱係数，あとで場所ごとに変わるように変更
	c = 0.225600000 #mm/ps
	D = 1 / (3*(1 - g)*myu_s) #光拡散係数
	n_rel = 1.33
	rd = -1.440/n_rel**2 + 0.710/n_rel + 0.668 + 0.0636*n_rel
	A = (1+rd) / (1-rd)
	myu_a_without = np.ones((stepnum_y,stepnum_x))*0.1 #吸収率，あとでnumpy arrayから持ってくるように変更
	myu_tmp = np.copy(myu_a_without)
	myu_tmp[20:25,20:25] = 100
	myu_a_with = myu_tmp
	x = np.linspace(0,length_x,stepnum_x)
	y = np.linspace(0,length_y,stepnum_y)
	stepnum_time = 220
	num_detector = 1
	pos_detector = np.array([[25,-1],[]])
	center_y = stepnum_y/2
	num_light = 1
	pos_light = np.array([[center_y,0],[]])

	def pulse(self,amp=10,t1=15,t2=5,dlen=stepnum_time,_dt=1):
		t = np.linspace(0,_dt*(dlen-1),dlen)
		y = amp * (np.exp(-((t-t1) ** 2)/(t2 ** 2)))
		print(y.shape)
		fig = plt.figure()
		fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
		bar1=ax1.plot(t,y)

		fig.savefig("./image/test/pulse.png")
		plt.clf()
		plt.close()
		return y


