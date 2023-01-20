import numpy as np



import dot_parameter_test10x10_2


##################################
#条件設定
##################################
#ピコ秒での計測
myClass = dot_parameter_test10x10_2.Dot()
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

H_j = np.zeros((num_detector,stepnum_x,stepnum_y))
intensity = int(1)

if stepnum_x%2==0:
	center = stepnum_x / 2
else:
	center = int((stepnum_x -1)/2)
first = center -3
end =center +3
test = np.array([first,center,end])
array = np.empty((stepnum_x - 3 + 6,2))
for i,x in enumerate(range(2,stepnum_x-1)):
	array[i,:] = [x,-2]
for y in [1,-2]:
	for x in test:
		i= i+1
		array[i,:] = [y,x]
print(array)
print(test)
