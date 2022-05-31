import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = np.maximum(0, x)

plt.figure(figsize=(10, 5))
plt.xlabel('net input z')
plt.ylabel('activation a')
plt.plot(x, y)
plt.show()