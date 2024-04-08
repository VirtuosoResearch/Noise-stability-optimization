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
1.611233541, 1.221935814, 1.080815683, 1.031370992, 1.02173367, 1.011333509, 1.0096493319
])

nso_std = np.array([0.013872, 0.010272, 0.012672, 0.011472, 0.011544, 0.009246,
       0.01272 ])

ours = np.array([
1.611233541, 1.057945567, 0.913238423, 0.889103688, 0.8712443596, 0.869834874, 0.86843861
])

ours_std = np.array([0.01818184, 0.012906, 0.01753682, 0.01820124, 0.0220384,
       0.01689508, 0.01305704])


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
ax.set_ylabel(r"$L(f_W)$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([0.75, 1.65]) 
plt.yticks(np.arange(0.8, 1.61, 0.4))

plt.xticks(np.arange(0, 7, 2), [r"$0$", r"$10$", r"$20$", r"$30$"])
#ax.set_title(r'${L(f_W)}$', fontsize=36)

#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.get_offset_text().set_fontsize(36)

# plt.gca().invert_xaxis()

# make the legend transparent
plt.legend(fontsize=30) # frameon=False)

ax.grid(ls=":", lw=0.4)
plt.tight_layout()
plt.savefig(f"./resnet_indoor_loss_l2_regularization.pdf", format="pdf", dpi=1200)
plt.show()

# %%
