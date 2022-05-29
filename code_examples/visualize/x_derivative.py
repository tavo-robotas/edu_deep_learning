import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def function(x):
    return 2*x
  
def deriv(x):
    return derivative(function, x)
  
y = np.linspace(0, 10) 

Yp = y[-10]
# plt.plot(Xp, Yp, marker='o')
# plt.vlines(Xp, min(Y), Yp, linestyles='dashed')
# plt.hlines(Yp, min(X), Xp, linestyles='dashed')
    
plt.figure(figsize=(10,8))
plt.plot(y, function(y), color='black', lw=3 ,label='f(x)=2x')
#plt.plot(y, deriv(y), color='green', label='Derivative')
plt.legend(loc='upper left')
plt.grid(True)