import itertools
import numpy as np

def test1(ac_time):
	index = 0
	H_j_array = np.zeros((ac_time.shape[0]))
	for t in range(10):
		if t in ac_time:
			H_j_array[index] = 10
			index = index+1
	print(H_j_array.shape)
	return H_j_array


accum_time_array = np.arange(0,10,step=3)
accum_time_array = np.delete(accum_time_array,0)
#ac_np_array = np.asarray(accum_time_array)
ac_np_array = accum_time_array
print(ac_np_array)
print(ac_np_array.shape)
all_num = itertools.product(range(2),range(2))
H_j = np.zeros((ac_np_array.shape[0],2,2))
print(H_j.shape)
for i, j in all_num:
	H_j[:,i,j] = test1(ac_np_array)

print(H_j.shape)
print(H_j[0,1])
print(H_j[1,1])
print(H_j[2,1])