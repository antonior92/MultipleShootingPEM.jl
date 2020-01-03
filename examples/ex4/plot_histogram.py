# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA


solutions = pd.read_csv('./narx_0.05_solutions.csv')
fig, ax = plt.subplots(nrows=3)
ax[0].hist(solutions['y[k-1]'])
ax[0].axvline(THETA[0])
ax[1].hist(solutions['y[k-2]'])
ax[1].axvline(THETA[1])
ax[2].hist(solutions['u[k-1]'])
ax[2].axvline(THETA[2])
plt.show()
