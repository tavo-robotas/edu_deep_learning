import numpy as np
import matplotlib.pyplot as plt
 
def objective(x):
    return x**2.0
 
r_min, r_max = -5.0, 5.0
inputs = np.arange(r_min, r_max, 0.1)
results = objective(inputs)
plt.figure(figsize=(5,4))
plt.plot(inputs, results)
plt.xlabel('w1', fontsize=18)
plt.ylabel('L', fontsize=18)
plt.xticks([])
plt.yticks([])
plt.show();