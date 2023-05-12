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

sgd = np.array([
0, 0.6279387474, 0.9780294895, 1.018746853, 1.055985451, 1.063967466, 1.065854907
])
sgd_std = np.array([
0, 0.00205598545, 0.003256345245, 0.003642233161, 0.003732676308, 0.00282072859, 0.003186974081
])
ours = np.array([
0, 0.4642417431, 0.7180556059, 0.7514595985, 0.7784501314, 0.781717062, 0.7860234976
])
# 0.599731052, 0.5755866234,  
ours_std = np.array([
0, 0.01042135473, 0.01613012723, 0.009875838425, 0.00852375003, 0.008721933781, 0.008830141747
])


# %%
f, ax = plt.subplots(figsize=(6,4.5)) 

x_axis = np.arange(len(sgd))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], sgd[i], s=80, marker="o", edgecolors = "none", facecolors='royalblue')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='forestgreen')

# ax.plot(x_axis, auc_clintox,  lw=3, color="darkblue", ls='solid')
# ax.plot(x_axis, auc_bbbp,  lw=3, color="darkred", ls='solid')

plt.errorbar(x_axis, sgd, linestyle='solid', lw=4, color="royalblue", label=r"$\mathrm{SGD}$")
plt.fill_between(
    x_axis, 
    sgd + sgd_std, 
    sgd - sgd_std, color="royalblue", alpha=0.3
)

plt.errorbar(x_axis, ours, linestyle='solid', lw=4, color="forestgreen", label=r"$\mathrm{NSO}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="forestgreen", alpha=0.3
)

ax.set_xlabel(r"$\mathrm{Number~of~Epochs}$", fontsize = 28)
ax.set_ylabel(r"$\mathrm{Generalization~Gap}$", fontsize = 28)
ax.tick_params(labelsize=28)
ax.set_ylim([-0.05, 1.25]) 
# plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])

# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
plt.yticks(np.arange(0., 1.21, 0.3),) # [r"$0$", "", r"$0.2$", "", r"$0.4$"]  
plt.xticks(np.arange(0, 7, 1), [r"$0$", r"$5$", r"$10$", r"$15$", r"$20$", r"$25$", r"$30$"])

ax.set_title(r'$\mathrm{CIFAR}$'+'-'+r'$\mathrm{100}$', fontsize=28)
# plt.gca().invert_xaxis()

plt.legend(fontsize=22)

ax.grid(lw=0.8)
plt.tight_layout()
plt.savefig(f"./resnet_cifar100_generalization_errs.pdf", format="pdf", dpi=1200)
plt.show()


# %%
