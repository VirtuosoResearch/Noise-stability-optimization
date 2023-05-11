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
l2 = np.array([554.8099224, 1286.976491, 1149.374658, 833.9042862, 1239.902828, 2966.033262, 1522.732252, 4870.373679]) 
l3 = np.array([354.4258618, 1061.699254, 986.9349023, 658.4348052, 958.7597881, 2675.085449, 1420.426826, 3419.376721])
l6 = np.array([322.7256736, 1045.689457, 620.5532369, 586.4944888, 612.4443558, 2454.170952, 1372.679497, 2184.593824])
l7 = np.array([338.4508594, 1004.683276, 616.8287567, 540.7129033, 567.0803631, 1942.138095, 672.4926358, 1015.488763])
l8 = np.array([319.2411059, 986.7543825, 611.7671177, 517.4672778, 550.3938508, 2007.985987, 614.6387021, 987.8733124])


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
#ax.set_ylabel(r'', fontsize=48)
ax.set_xticks(ind + width  + shift + 4)
ax.set_xticklabels(msa_name_list, fontsize=40)
# plt.yticks(np.arange(0, 12500, 2000))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(40)
plt.yticks(np.arange(0, 4001, 1000))
plt.ylim([0, 4000])

plt.tick_params(axis='x')


ax.legend(
    (rects7[0], rects6[0], rects3[0], rects2[0]), 
    (r'$\mathrm{NSO}~(k=2)$', r'$\mathrm{NSO}~(k=1)$', r'$\mathrm{SAM}$', r'$\mathrm{SGD}$'), 
    loc=2, fontsize=40, ncol=2)

ax.yaxis.grid(True, lw=0.4)
ax.set_title('$\lambda_1[\mathbf{H}]$', fontsize=48,  y=1.02)

ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('comparsion_hessian_eigenvalues.pdf', format='pdf', dpi=100)
