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

msa_name_list = ['Indoor', 'Caltech-252', 'Aircrafts', 'CIFAR-10', 'CIFAR-100']


l2 = np.array([0.7529, 0.6342, 0.6226, 0.1087, 0.550108]) # 14372.860
l3 = np.array([0.7297, 0.5711, 0.5879, 0.1026, 0.5379]) # 10230.431
l6 = np.array([0.639, 0.5343, 0.4521, 0.0948, 0.5140])

# l2 = np.log10(l2)
# l3 = np.log10(l3)
# l6 = np.log10(l6)
# l7 = np.log10(l7)

N = 5
ind = np.arange(N) * 24  # the x locations for the groups
width = 4.0      # the width of the bars
shift = 0.8


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(18,6.5))
#rects8 = ax.bar(ind + shift * 0, l7, width, color='crimson', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
# rects7 = ax.bar(ind + width * 1 + shift, l7, width, color='forestgreen', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
rects6 = ax.bar(ind + width * 2 + shift*2, l6, width, color='royalblue', ecolor='white') # color='tomato', ecolor='k', hatch="x"
rects3 = ax.bar(ind + width * 3 + shift*3, l3, width, color='orange', ecolor='white')
rects2 = ax.bar(ind + width * 4 + shift*4, l2, width, color='lightgrey', ecolor='white')

ax.set_ylim([0, 0.87])
ax.set_yticks(np.arange(0, 0.85, 0.2))
ax.set_ylabel('Test loss', fontsize=52)
ax.set_xticks(np.array([10.3, 33.3, 58.3, 81.3, 107.3]))# ind + width  + shift + 7.5
ax.set_xticklabels(msa_name_list, fontsize=52)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(40)

# plt.yticks(np.arange(0, 12001, 3000))
# plt.ylim([0, 12001])

plt.tick_params(axis='x')


# ax.legend(
#     (rects6[0], rects3[0], rects2[0]), 
#     (r'$\mathrm{NSO}$', r'$\mathrm{SAM}$', r'$\mathrm{WP-SGD}$'), 
#     loc=1, fontsize=40, ncol=3)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Test~loss}$', fontsize=48, x=0.474, y=1.02)


ax.tick_params(axis='both', which='major', labelsize=52)
ax.tick_params(axis='both', which='minor', labelsize=52)

plt.tight_layout()
plt.savefig('comparison_test_losses.pdf', format='pdf', dpi=100)
#plt.show()
