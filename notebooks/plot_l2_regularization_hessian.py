# %%
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.stats import pearsonr, spearmanr

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

nso = np.array([
8121.40447, 5643.65056, 4521.40447, 3562.670991, 3262.716155, 3179.045358, 3151.054327,
])

nso_std = np.array([  8.18182435, 139.9986492 , 122.50471515, 127.5599978 ,
        71.7411387 ,  72.51687085,  59.99406125])

ours = np.array([
8121.40447, 4663.682319, 3626.522782, 2862.243672, 2761.423283, 2756.953663, 2743.986471,
])

ours_std = np.array([16.3636487 , 69.62484955, 68.94553085, 75.81766235, 91.40611805,
       69.97910645, 54.9483926 ])


f, ax = plt.subplots(figsize=(7,6)) 

x_axis = np.arange(len(nso))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], nso[i], s=80, marker="o", edgecolors = "none", facecolors='black')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='black')

# ax.plot(x_axis, auc_clintox,  lw=3, color="darkblue", ls='solid')
# ax.plot(x_axis, auc_bbbp,  lw=3, color="darkred", ls='solid')

plt.errorbar(x_axis, nso, linestyle='--', lw=4, color="k", label=r"$\mathrm{w/o~dist.~reg.}$")
plt.fill_between(
    x_axis, 
    nso + nso_std, 
    nso - nso_std, color="k", alpha=0.3
)

plt.errorbar(x_axis, ours, linestyle='solid', lw=4, color="r", label=r"$\mathrm{w/~dist.~reg.}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="r", alpha=0.3
)

ax.set_xlabel(r"$t$", fontsize = 36)
ax.set_ylabel(r"$\mathrm{Trace}$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([2300, 8500]) 
plt.yticks(np.arange(3000, 8001, 2000))

plt.xticks(np.arange(0, 7, 2), [r"$0$", r"$10$", r"$20$", r"$30$"])
#ax.set_title(r'$\mathrm{Hessian~trace}$' + r' $(\times$' + r'$10^3)$', fontsize=36)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(36)

# plt.gca().invert_xaxis()

plt.legend(fontsize=30)

ax.grid(ls=":", lw=0.4)
plt.tight_layout()
plt.savefig(f"./resnet_indoor_hessia_trace_l2_regularization.pdf", format="pdf", dpi=1200)
plt.show()

# %%
