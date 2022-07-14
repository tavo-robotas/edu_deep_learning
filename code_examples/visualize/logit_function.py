import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
      return 1 / (1 + np.exp(-z))
    
def logit(p):
    return np.log(p) - np.log(1 - p)

xs = np.linspace(10, -6)
ac = sigmoid(xs)
ys = logit(ac)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.set_dpi(100)
ax1.plot(xs, ac)
ax1.grid(True)
ax1.set_title('sigmoid')
ax1.set_xlabel('z', fontsize=12)
ax1.set_ylabel('a', fontsize=12)
ax2.plot(ac, ys)
ax2.set_xlabel('p', fontsize=12)
ax2.set_ylabel('log(p / (1 - p)) = z', fontsize=12)
ax2.set_title('logit')
ax2.grid(True)