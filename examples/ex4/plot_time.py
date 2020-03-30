# %%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{nicefrac}')
#matplotlib.verbose.level = 'debug-annoying'

time_const_list = ['fast', 'interm', 'slow']
list_n_stage = [3, 5, 7, 10, 20]
list_shoot_len = [2, 5, 10, 20]
noise_std = 0.05
folder = './solutions/'
q1 = 24
q2 = 49
q3 = 74

# %% multiple shooting

tp = [folder + 'multipleshoot_cp_100_fev/multipleshoot_{}'.format(shoot_len) for shoot_len in list_shoot_len]

cols = ['{}{}_{}'.format(nm, tp, time_const) for nm in ['nfev', 'time_per_iter']
        for tp in ['_lb', '', '_ub'] for time_const in time_const_list]
df = pd.DataFrame(np.zeros((len(tp), 18)), columns=cols, index=list_shoot_len)
for time_const in time_const_list:
    for i, t in enumerate(tp):
        solutions = pd.read_csv('' + t + '_' + str(noise_std) + '_' + time_const + '_solutions.csv')
        xx = np.array(solutions['exec time'] / solutions['nfev'])
        xx = np.sort(xx)
        df['time_per_iter_lb_{}'.format(time_const)].iloc[i] = xx[q1]
        df['time_per_iter_{}'.format(time_const)].iloc[i] = xx[q2]
        df['time_per_iter_ub_{}'.format(time_const)].iloc[i] = xx[q3]

        xx = np.array(solutions['nfev'])
        xx = np.sort(xx)
        df['nfev_lb_{}'.format(time_const)].iloc[i] = xx[q1]
        df['nfev_{}'.format(time_const)].iloc[i] = xx[q2]
        df['nfev_ub_{}'.format(time_const)].iloc[i] = xx[q3]

print('multipleshoot')
print(df)
df.to_csv('multipleshoot_percentiles')


# %% multistage

tp = [folder + 'multistage_fev/multistage_{}'.format(n_stage) for n_stage in list_n_stage]
cols = ['{}{}_{}'.format(nm, tp, time_const) for nm in ['nfev', 'time_per_iter']
        for tp in ['_lb', '', '_ub'] for time_const in time_const_list]
df_multistage = pd.DataFrame(np.zeros((len(tp), 18)), columns=cols, index=list_n_stage)
for time_const in time_const_list:
    for i, t in enumerate(tp):
        solutions = pd.read_csv('' + t + '_' + str(noise_std) + '_' + time_const + '_solutions.csv')
        xx = np.array(solutions['exec time'] / solutions['nfev'])
        xx = np.sort(xx)
        df_multistage['time_per_iter_lb_{}'.format(time_const)].iloc[i] = xx[q1]
        df_multistage['time_per_iter_{}'.format(time_const)].iloc[i] = xx[q2]
        df_multistage['time_per_iter_ub_{}'.format(time_const)].iloc[i] = xx[q3]

        xx = np.array(solutions['nfev'])
        xx = np.sort(xx)
        df_multistage['nfev_lb_{}'.format(time_const)].iloc[i] = xx[q1]
        df_multistage['nfev_{}'.format(time_const)].iloc[i] = xx[q2]
        df_multistage['nfev_ub_{}'.format(time_const)].iloc[i] = xx[q3]

print('nstage')
print(df_multistage)
df_multistage.to_csv('multistage_percentiles')


# %% plot

fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(8,5))
colorss = ['b', 'r', 'g']
for i, ddf in enumerate([df, df_multistage]):
    for j, tp in enumerate(['time_per_iter', 'nfev']):
        for kk, time_const in enumerate(time_const_list):
            cc = colorss[kk]
            ax[j, i].plot(ddf.index, ddf[tp + '_' + time_const], color=cc, label=time_const)
            ax[j, i].plot(ddf.index, ddf[tp + '_lb' + '_' + time_const], color=cc, alpha=0.2, ls='--')
            ax[j, i].plot(ddf.index, ddf[tp + '_ub' + '_' + time_const], color=cc, alpha=0.2, ls='--')
            ax[j, i].fill_between(ddf.index, ddf[tp + '_lb' + '_' + time_const],
                          ddf[tp + '_ub' + '_' + time_const], color=cc, alpha=0.2)

ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
#ax[0, 0].set_title('multiple shoot')
#ax[0, 1].set_title('multistage')
ax[0, 0].set_ylabel(r'$\nicefrac{t_{\rm run}}{n_{\rm ev}}$', rotation=0, fontsize=20)
ax[0, 0].yaxis.set_label_coords(-0.17 ,0.4)
ax[1, 0].set_ylabel(r'$n_{\rm ev}$', rotation=0, fontsize=15, labelpad=15)
ax[1, 0].yaxis.set_label_coords(-0.17 ,0.45)
ax[1, 0].set_xlabel(r'$\Delta {\rm m}_{\max}$', fontsize=12)
ax[1, 1].xaxis.set_label_coords(0.5, -0.14)
ax[1, 1].set_xlabel(r'$K$', fontsize=15)
ax[0, 0].legend()

# Save
extent = ax[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('ex4_running_time_1.pdf', bbox_inches=extent.expanded(1.4, 1.1).translated(-0.4, 0))
extent = ax[0, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('ex4_running_time_2.pdf', bbox_inches=extent.expanded(1.15, 1.1).translated(-0.15, 0))
extent = ax[1, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('ex4_running_time_3.pdf', bbox_inches=extent.expanded(1.4, 1.3).translated(-0.4, -0.2))
extent = ax[1, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('ex4_running_time_4.pdf', bbox_inches=extent.expanded(1.15, 1.3).translated(-0.15, -0.2))

plt.show()