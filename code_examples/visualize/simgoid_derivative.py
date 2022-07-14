import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return np.exp(-z)/(1 + np.exp(-z))**2

x = np.linspace(-8, 8)
y = sigmoid(x)

plt.figure(figsize=(12,6))
plt.axvline(x=0 , c='r', ls='--' )
#plt.axhline(y=0.5 , c='b', ls='-.')
plt.grid(True)
plt.plot(x, y);