# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA

type = ['noe', 'multistage_2', 'multistage_5', 'narx']

fig, ax = plt.subplots(nrows=3)

for t in type:
    solutions = pd.read_csv('./'+ t + '_0.05_solutions.csv')
    ax[0].hist(solutions['y[k-1]'], label=t)
    ax[1].hist(solutions['y[k-2]'], label=t)
    ax[2].hist(solutions['u[k-1]'], label=t)

ax[0].axvline(THETA[0])
ax[1].axvline(THETA[1])
ax[2].axvline(THETA[2])
plt.legend()
plt.show()
