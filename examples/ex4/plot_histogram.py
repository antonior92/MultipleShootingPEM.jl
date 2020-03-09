# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA1, THETA2, THETA3

time_const = 'slow'
list_n_stage = [2, 3, 4, 5, 6, 7, 9, 10, 20]
list_shoot_len = [2, 5, 10, 20]
lateral = 0.
noise_std = 0.05
folder = './solutions/'

if time_const == 'interm':
    THETA = THETA1
if time_const == 'fast':
    THETA = THETA2
elif time_const == 'slow':
    THETA = THETA3

vars = ['y[k-1]', 'y[k-2]', 'u[k-1]']
tp = []
tp += [folder + 'multipleshoot_cp_100/multipleshoot_{}'.format(shoot_len) for shoot_len in list_shoot_len]
tp += [folder + 'multistage/multistage_{}'.format(n_stage) for n_stage in list_n_stage]
bins = 150
rg = 0.015
thres = 0.5 *rg
logscale = 500
fig, ax = plt.subplots(nrows=3)

for t in tp:
    solutions = pd.read_csv('' + t + '_'+ str(noise_std) + '_' + time_const + '_solutions.csv')

    for i, v in enumerate(vars):
        normalized_value = solutions[vars[i]] - THETA[i]
        two_scales = np.zeros_like(normalized_value)
        for j in range(len(normalized_value)):
            if normalized_value[j] > thres:
                two_scales[j] = np.log(normalized_value[j]-thres)/logscale + thres
            elif thres > normalized_value[j] > -thres:
                two_scales[j] = normalized_value[j]
            else:
                two_scales[j] = -np.log(thres-normalized_value[j])/logscale - thres
        ax[i].hist(two_scales, label=t, bins=bins, range=[-rg, rg])
        ax[i].set_xlim([-rg, rg])
        ax[i].axvline(thres)
        ax[i].axvline(0)
        ax[i].axvline(-thres)


plt.show()
