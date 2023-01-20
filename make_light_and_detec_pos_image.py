import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import dot_parameter_test32x32_2


##################################
#条件設定
##################################
#ピコ秒での計測
myClass = dot_parameter_test32x32_2.Dot()
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
myu_a_with = myClass.myu_a_with
x = myClass.x
y = myClass.y
stepnum_time = myClass.stepnum_time 
accum_time = myClass.accum_time 
accum_time_array = np.arange(stepnum_time,step=accum_time)
accum_time_array = np.delete(accum_time_array,0)

intensity = int(1)

num_detector = myClass.num_detector
pos_detector = myClass.pos_detector

num_light = myClass.num_light
pos_light = myClass.pos_light
##################################


for i in range(num_light):
	
	myu_a_with[pos_light[i,0],pos_light[i,1]] = 0.02

#for j in range(num_detector):
	#myu_a_with[pos_detector[j,0],pos_detector[j,1]] = 0.02

fig = plt.figure()
fig, ax1= plt.subplots(1, 1, figsize=(8, 4.5),sharex=True, sharey=True)
#bar1=ax1.imshow(x_reshape, cmap=cm.Greys,vmin=0,vmax=0.02)
bar1=ax1.imshow(myu_a_with, cmap=cm.Greys,vmin=0)
fig.colorbar(bar1)
#plt.show()
fig.savefig("./image/myu_a_with_light_detecter-2.png")
plt.close(fig)

print(num_light)
print(num_detector)