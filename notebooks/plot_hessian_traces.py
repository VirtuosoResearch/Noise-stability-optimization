# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

msa_name_list = ['CIFAR-10', 'CIFAR-100', 'Messidor', 'Aircrafts', 'Flowers', 'Birds', 'Caltech-256', 'Cars']

# fill in data
l2 = np.array([4738.039, 14372.860, 5555.625, 6218.884, 1645.552057, 4252.13313, 4078.692163, 5406.93864]) 
l3 = np.array([2965.668, 10230.431, 5079.884, 5034.589, 1154.878304, 3515.340279, 3789.30885, 4379.775769])
l6 = np.array([2521.030, 6124.207, 4975.861, 4502.387, 963.1627191, 3067.799648, 3264.214885, 3473.67205])
l7 = np.array([2247.021148, 5922.739259, 4429.547208, 3910.529684, 866.0727367, 2930.410851, 3024.123527, 3319.434227])
l8 = np.array([2087.463558, 5829.863954, 3964.895751, 3850.838496, 837.1619756, 2804.22605, 2910.317867, 3227.13439])


# l2 = np.log10(l2)
# l3 = np.log10(l3)
# l6 = np.log10(l6)
# l7 = np.log10(l7)

N = 8
ind = np.arange(N) * 24  # the x locations for the groups
width = 3.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(30,6))
rects8 = ax.bar(ind + shift * 0, l7, width, color='crimson', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
rects7 = ax.bar(ind + width * 1 + shift, l7, width, color='royalblue', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
rects6 = ax.bar(ind + width * 2 + shift*2, l6, width, color='orange', ecolor='white') # color='tomato', ecolor='k', hatch="x"
rects3 = ax.bar(ind + width * 3 + shift*3, l3, width, color='forestgreen', ecolor='white')
rects2 = ax.bar(ind + width * 4 + shift*4, l2, width, color='lightgrey', ecolor='white')

#ax.set_ylim([0.2, 400])
ax.set_ylabel('Trace value', fontsize=36)
ax.set_xticks(ind + width  + shift + 4)
ax.set_xticklabels(msa_name_list, fontsize=16)
plt.yticks(np.arange(0, 12500, 2000))
plt.ylim([0, 12000])

plt.tick_params(axis='x')


ax.legend(
    (rects8[0], rects7[0], rects6[0], rects3[0], rects2[0]), 
    (r'$\mathrm{Noise~Stability~Optimization}~(k=3)$', r'$\mathrm{Noise~Stability~Optimization}~(k=2)$', r'$\mathrm{Noise~Stability~Optimization}~(k=1)$', r'$\mathrm{Sharpness~Aware~Minimization}$', r'$\mathrm{Stochastic~Gradient~Descent}$'), 
    loc=1, fontsize=28, ncol=3)

ax.yaxis.grid(True, lw=0.4)
ax.set_title('Trace of the Weight Hessian Matrix', fontsize=36,  y=1.02)

ax.tick_params(axis='both', which='major', labelsize=36)
ax.tick_params(axis='both', which='minor', labelsize=36)

plt.tight_layout()
plt.savefig('comparsion_hessian_traces.pdf', format='pdf', dpi=100)