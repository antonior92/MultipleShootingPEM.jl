# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA1, THETA2, THETA3

time_const = 'interm'
list_n_stage = [2, 3, 7, 8, 9, 10]

if time_const == 'interm':
    THETA = THETA1
    bins = None
if time_const == 'fast':
    THETA = THETA2
    bins = None
elif time_const == 'slow':
    THETA = THETA3
    bins = 100

type = ['multistage_{}'.format(n_stage) for n_stage in list_n_stage]

fig, ax = plt.subplots(nrows=3)

for t in type:
    solutions = pd.read_csv('./solutions/'+ t + '_0.05_' + time_const + '_solutions.csv')
    ax[0].hist(solutions['y[k-1]'], label=t, bins=bins)
    ax[1].hist(solutions['y[k-2]'], label=t, bins=bins)
    ax[2].hist(solutions['u[k-1]'], label=t, bins=bins)

ax[0].axvline(THETA[0])
ax[1].axvline(THETA[1])
ax[2].axvline(THETA[2])
plt.legend()
plt.show()
