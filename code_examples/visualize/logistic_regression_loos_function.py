from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def loss(y , a):
    return (-y * np.log(a)) + (1 - y) * np.log(1 - a)

xs = [sigmoid(x) for x in np.linspace(-8, 8)]
ys_0 = [-loss(0, x) for x in xs ]
ys_1 = [ loss(1, x) for x in xs ]
plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(xs, ys_1, label='1')
plt.plot(xs, ys_0, label='0')
plt.xlabel('a', fontsize=14)
plt.ylabel('L', fontsize=14)
plt.legend(loc='upper left')