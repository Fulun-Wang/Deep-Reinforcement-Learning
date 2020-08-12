import matplotlib.pyplot as plt
import numpy as np

map = np.full((400, 400), int(10), dtype=np.int8)
map[23, 25] = 2
map[372, 367] = 5
map[60:70, 15:64] = 0
map[10:35, 46:54] = 0
map[42:54, 78:86] = 0
map[18:27, 80:120] = 0
map[53:116, 110:116] = 0
map[78:169, 75:92] = 0
map[147:154, 100:167] = 0
map[87:95, 112:134] = 0
map[110:118, 133:186] = 0
map[137:187, 110:119] = 0
map[200:249, 110:119] = 0
map[52:259, 201:210] = 0
map[187:249, 139:179] = 0
map[277:283, 21:239] = 0
map[250:299, 270:275] = 0
map[189:229, 257:305] = 0
map[157:162, 219:309] = 0
map[320:369, 65:119] = 0
map[330:358, 139:259] = 0
map[346:355, 256:323] = 0
map[220:349, 343:346] = 0
map[287:289, 270:326] = 0
map[257:259, 318:376] = 0
map[277:330, 167:172] = 0
init_state = map[7:39, 9:41]
start_position = np.argwhere(map == 2)
plt.imshow(map, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
plt.colorbar()
x_ticks = np.arange(0, 400, 20)
y_ticks = np.arange(0, 400, 20)
x = np.array([23, 25, 26])
y = np.array([24, 26, 78])
print(x, y)
plt.xlabel('y')
plt.ylabel('x')
plt.xticks(x_ticks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                     '14', '15', '16', '17', '18', '19', '20'), color = 'blue')
plt.yticks(y_ticks, ('20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8',
                     '7', '6', '5', '4', '3', '2', '1'), color = 'blue')
plt.grid(0)
plt.plot(x, y)
plt.show()
