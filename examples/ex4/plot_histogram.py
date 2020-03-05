# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA1, THETA2, THETA3

time_const = 'slow'
list_n_stage = [2, 3, 4, 5, 6, 7, 9, 10]
list_shoot_len = [2, 5, 10, 20]
lateral = 4.0
noise_std = 0.02
folder = './solutions/'

if time_const == 'interm':
    THETA = THETA1
if time_const == 'fast':
    THETA = THETA2
elif time_const == 'slow':
    THETA = THETA3


tp = []
tp += [folder + 'multipleshoot_cp_100/multipleshoot_{}'.format(shoot_len) for shoot_len in list_shoot_len]
#tp += [folder + 'multistage/multistage_{}'.format(n_stage) for n_stage in list_n_stage]
#r = [[THETA[i] - lateral, THETA[i] + lateral] for i in range(3)]
bins = 30
fig, ax = plt.subplots(nrows=3)
for t in tp:
    solutions = pd.read_csv('' + t + '_'+ str(noise_std) + '_' + time_const + '_solutions.csv')

    ax[0].hist(solutions['y[k-1]'], label=t, bins=bins)#, range=r[0])
    ax[1].hist(solutions['y[k-2]'], label=t, bins=bins)#, range=r[1])
    ax[2].hist(solutions['u[k-1]'], label=t, bins=bins)#, range=r[2])

for i in range(2):
    ax[i].axvline(THETA[i])
    #ax[i].set_xlim(r[i])
#plt.legend()
plt.show()
