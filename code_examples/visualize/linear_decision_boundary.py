import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

da = pd.read_csv("../data_samples/cells.csv")
x, y = da.cell_size, da.malignant
y_p = np.array(
    [
        0.04824502, 0.08739737, 0.12654972, 0.24400676, 0.26358293,
        0.28315911, 0.30273528, 0.32231145, 0.3614638 , 0.51807319,
        0.55722554, 0.57680171, 0.59637788, 0.67468258, 0.69425875,
        0.79213962, 0.89002049, 0.98790136, 1.67306743
    ]
)
plt.figure(figsize=(12,5))
plt.xlabel('feature x', fontsize=14)
plt.ylabel('class label', fontsize=14)
plt.axhline(y=0.5, color='black', linestyle='--')
plt.xlim(10, 60)
plt.ylim(-0.2, 1.2)
plt.yticks(np.arange(2), ['0', 1])
plt.scatter(x, y, marker='+',s=100, c=y);
plt.plot(x, y_p)
plt.grid(True)