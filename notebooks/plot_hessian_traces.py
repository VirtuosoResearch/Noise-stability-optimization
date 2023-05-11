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

msa_name_list = ['Flowers', 'Birds', 'Caltech-256', 'Cars', 'Aircrafts', 'Ophthalmology', 'CIFAR-10', 'CIFAR-100']

# fill in data
l2 = np.array([1645.552057, 4252.13313, 4078.692163, 5406.93864, 6218.884, 5555.625, 4738.039, 14372.860]) # 14372.860
l3 = np.array([1154.878304, 3515.340279, 3789.30885, 4379.775769, 5034.589, 5079.884, 2965.668, 10230.431]) # 10230.431
l6 = np.array([963.1627191, 3067.799648, 3264.214885, 3473.67205, 4502.387, 4975.861, 2521.030, 6124.207])
l7 = np.array([866.0727367, 2930.410851, 3024.123527, 3319.434227, 3910.529684, 4429.547208, 2247.021148, 5922.739259])
l8 = np.array([837.1619756, 2804.22605, 2910.317867, 3227.13439, 3964.895751, 3850.838496, 2087.463558, 5829.863954])


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
#rects8 = ax.bar(ind + shift * 0, l7, width, color='crimson', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
rects7 = ax.bar(ind + width * 1 + shift, l7, width, color='royalblue', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
rects6 = ax.bar(ind + width * 2 + shift*2, l6, width, color='orange', ecolor='white') # color='tomato', ecolor='k', hatch="x"
rects3 = ax.bar(ind + width * 3 + shift*3, l3, width, color='forestgreen', ecolor='white')
rects2 = ax.bar(ind + width * 4 + shift*4, l2, width, color='lightgrey', ecolor='white')

#ax.set_ylim([0.2, 400])
#ax.set_ylabel('Trace value', fontsize=36)
ax.set_xticks(ind + width  + shift + 4)
ax.set_xticklabels(msa_name_list, fontsize=40)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 12001, 2000))
plt.ylim([0, 12001])

plt.tick_params(axis='x')


#ax.legend(
#    (rects8[0], rects7[0], rects6[0], rects3[0], rects2[0]), 
#    (r'$\mathrm{Noise~Stability~Optimization}~(k=3)$', r'$\mathrm{Noise~Stability~Optimization}~(k=2)$', r'$\mathrm{Noise~Stability~Optimization}~(k=1)$', r'$\mathrm{Sharpness~Aware~Minimization}$', r'$\mathrm{Stochastic~Gradient~Descent}$'), 
#    loc=1, fontsize=28, ncol=3)
ax.legend(
    (rects7[0], rects6[0], rects3[0], rects2[0]), 
    (r'$\mathrm{NSO}~(k=2)$', r'$\mathrm{NSO}~(k=1)$', r'$\mathrm{SAM}$', r'$\mathrm{SGD}$'), 
    loc=2, fontsize=40, ncol=2)



ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\textup{Tr}[\mathbf{H}]$', fontsize=48, x=0.474, y=1.02)

ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparsion_hessian_traces.pdf', format='pdf', dpi=100)
#plt.show()
