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

ours = np.array([
6124.207,
5922.739259,
5829.863954,
5802.234,
])

ours_std =  np.array([40.62494135, 31.10182508, 24.42150783, 20.36340549])

# %%
f, ax = plt.subplots(figsize=(6,4)) 

x_axis = np.arange(len(ours))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='royalblue')

plt.errorbar(x_axis, ours, linestyle='solid', lw=6, color="royalblue", label=r"$\mathrm{NSO}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="royalblue", alpha=0.3
)

ax.set_xlabel(r"$k$", fontsize = 32)
ax.set_ylabel(r'$\textup{Tr}[\mathbf{H}]$', fontsize = 32)
ax.tick_params(labelsize=32)
ax.set_ylim([5700, 6500]) 
# plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])

# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(40)
# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
plt.yticks(np.arange(5800, 6500, 150), [r"$5800$", "", r"$6100$", "", r"$6400$"])
plt.xticks(np.arange(4), [r"$1$", r"$2$", r"$3$", r"$4$"])

# plt.gca().invert_xaxis()

# plt.legend(fontsize=22)

ax.grid(lw=0.8)
plt.tight_layout()
plt.savefig(f"./ablate_k_traces_cifar100.pdf", format="pdf", dpi=1200)
plt.show()