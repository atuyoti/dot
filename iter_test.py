import itertools
import numpy as np

import dot_parameter

H_map = np.zeros((11,11))
stepnum_time = 20
H_j = 0

def diffuse(nt,y,x,t_d):
	global H_j
	H_j = H_j+1
	print("nt:"+str(nt))

all_num = itertools.product(range(11),range(11))
for i, j in all_num:
	for t_d in range(stepnum_time):
		for t in range(stepnum_time):
			diffuse(t,i,j,t_d)
		print("aaaaaaaaaaaaaaaaa")