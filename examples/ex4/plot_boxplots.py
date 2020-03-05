# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import THETA1, THETA2, THETA3
import seaborn as sns

THETA = THETA1

folder = 'solutions/'
list_n_stage = [2, 3, 4, 5, 6, 7, 9, 10]
list_shoot_len = [2, 5, 10, 20]
time_const = ['interm', 'fast', 'slow']
tp = []
#tp += [folder + 'multipleshoot_cp_100/multipleshoot_{}'.format(shoot_len) for shoot_len in list_shoot_len]
tp += [folder + 'multistage/multistage_{}'.format(n_stage) for n_stage in list_n_stage]
#tp += ['/narx', '/noe']
noise_std = 0.05
bins = 100
solutions = pd.concat(
    [pd.concat([pd.read_csv(t + '_' + str(noise_std)+ '_' + tc + '_solutions.csv') for t in tp], keys=tp, names=['type'])
    for tc in time_const], keys=time_const, names=['time_const'])
solutions.reset_index(level=(0, 1), inplace=True)

solutions = solutions[solutions['time_const'] == 'interm']

fig, ax = plt.subplots(nrows=1)
sns.boxplot(data=solutions, x='type', y='y[k-1]', ax=ax)
ax.set_ylim([1.45, 1.55])
for tick in ax.get_xticklabels():
    tick.set_rotation(70)
plt.show()