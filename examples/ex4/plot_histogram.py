# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA1, THETA2, THETA3

time_const = 'slow'
list_n_stage = [2, 3, 4, 5, 7, 9, 10, 20]
list_shoot_len = [2, 5, 10, 20]
lateral = 0.
noise_std = 0.05
folder = './solutions/'

vars = ['y[k-1]', 'y[k-2]', 'u[k-1]']
tp = []
tp += [[folder + 'narx/narx'] + [folder + 'noe/noe']]
tp += [[folder + 'multipleshoot_cp_100/multipleshoot_{}'.format(shoot_len) for shoot_len in list_shoot_len]]
tp += [[folder + 'multistage/multistage_{}'.format(n_stage) for n_stage in list_n_stage]]
bins = 50
rg = 0.015
thres = 0.5 *rg
logscale = 200

for kk, v in enumerate(vars):
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True)
    for ll, time_const in enumerate(['fast', 'interm', 'slow']):
        if time_const == 'interm':
            THETA = THETA1
        if time_const == 'fast':
            THETA = THETA2
        elif time_const == 'slow':
            THETA = THETA3
        for i, tt in enumerate(tp):
            for t in tt:
                solutions = pd.read_csv('' + t + '_'+ str(noise_std) + '_' + time_const + '_solutions.csv')

                normalized_value = solutions[v] - THETA[kk]
                two_scales = np.zeros_like(normalized_value)
                for j in range(len(normalized_value)):
                    if normalized_value[j] > thres:
                        two_scales[j] = np.log(normalized_value[j]-thres + 1)/logscale + thres
                    elif thres > normalized_value[j] > -thres:
                        two_scales[j] = normalized_value[j]
                    else:
                        two_scales[j] = -np.log(thres-normalized_value[j] + 1)/logscale - thres
                ax[i, ll].hist(two_scales, label=t, bins=bins, range=[-rg, rg])
                ax[i, ll].set_xlim([-rg, rg])
                ax[i, ll].axvline(thres, ls='--')
                ax[i, ll].axvline(0, ls='--')
                ax[i, ll].axvline(-thres, ls='--')

    plt.tight_layout()
    plt.show()
