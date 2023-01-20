# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import os

class Dot:
	stepnum_x = 34
	stepnum_y = 34
	length_x = 1.32
	length_y = 1.32
	dx = length_x / (stepnum_x - 1)
	dy = length_y / (stepnum_y - 1)
	dt = 0.004 #ps 0.002
	g = 0 #非等方散乱パラメータ，あとで場所ごとに変わるように変更
	myu_s = 15 #散乱係数，あとで場所ごとに変わるように変更
	c = 0.225600000 #mm/ps
	D = 1 / (3*(1 - g)*myu_s) #光拡散係数
	n_rel = 1.33
	rd = -1.440/n_rel**2 + 0.710/n_rel + 0.668 + 0.0636*n_rel
	A = (1+rd) / (1-rd)
	myu_a_without = np.ones((stepnum_x,stepnum_y))*0.01
	myu_tmp = np.copy(myu_a_without)
	myu_tmp[9:14,9:14] = 0.02
	myu_tmp[22:27,22:27] = 0.02
	myu_tmp[9:14,22:27] = 0.015
	myu_tmp[22:27,9:14] = 0.015
	#myu_tmp[5:7,5:7] = 50
	myu_a_with = myu_tmp
	x = np.linspace(0,length_x,stepnum_x)
	y = np.linspace(0,length_y,stepnum_y)
	stepnum_time = 3501
	accum_time = 3500
	accum_time_array = np.arange(stepnum_time,step=accum_time)
	accum_time_array = np.delete(accum_time_array,0)
	"""
	if stepnum_x%2==0:
		center = stepnum_x / 2
	else:
		center = int((stepnum_x -1)/2)
	"""
	first = 3
	end = stepnum_y - 3
	num_detector_tate = len(list(range(first,end,5)))*2
	#print(num_detector_tate)
	pos_detector = np.empty((stepnum_x - 3 + num_detector_tate,2),dtype=int)
	for i,x in enumerate(range(2,stepnum_x-1)):
		pos_detector[i,:] = [x,-2]
	for y in [1,-2]:
		for x in range(first,end,5):
			i= i+1
			pos_detector[i,:] = [y,x]
	#pos_detector = np.array([[1,5],[1,8],[-2,5],[-2,8],[2,-2],[3,-2],[4,-2],[5,-2],[6,-2],[7,-2],[8,-2],[9,-2]])
	num_detector = pos_detector.shape[0]
	
	pos_light = np.empty((stepnum_x - 10,2),dtype=int)
	for i,x in enumerate(range(5,stepnum_x-5)):
		pos_light[i,:] = [x,0]
	num_light = pos_light.shape[0]
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

