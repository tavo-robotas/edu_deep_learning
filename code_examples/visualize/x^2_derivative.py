import matplotlib.pyplot as plt
import numpy as np
 
x1 = np.arange(-10, 10+0.1, 1)
y1 = x1**2
y2 = 2*x1
 
crosspoint_x = np.argwhere(np.sign(np.round(y1 - y2, 3)) == 0)
 
fig, ax = plt.subplots(figsize=(16,8))

ax.plot(x1, y1, color='red',  linewidth='1', label='y = x^2')
ax.plot(x1, y2, color='blue', linewidth='1', label='y = 2x')
ax.plot(x1[crosspoint_x], y1[crosspoint_x], 'o',  color='black')
 
ax.set_xticks(np.arange(-10,11,1))
ax.set_xticks(np.arange(-10,11,0.5) ,minor=True)
ax.grid(color='#bbbbff', linestyle='dashed', which="major")
ax.grid(color='#ddddff', linestyle='dotted', which="minor")
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
 
plt.show()