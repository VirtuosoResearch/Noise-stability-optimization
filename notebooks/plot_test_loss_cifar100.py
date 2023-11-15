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

sgd = np.array([0.852661, 0.705674, 0.5994  , 0.563422, 0.550002, 0.550922,
       0.550108])
sgd_std = np.array([0.006936, 0.005136, 0.006336, 0.005736, 0.005772, 0.004623,
       0.00636 ])


ours = np.array([0.859271, 0.678779, 0.555891, 0.527813, 0.516742, 0.516086,
       0.514034])
ours_std = np.array([0.005874, 0.005148, 0.005184, 0.004596, 0.004896, 0.004656,
       0.00516 ])


f, ax = plt.subplots(figsize=(6,5)) 

x_axis = np.arange(len(sgd))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], sgd[i], s=80, marker="o", edgecolors = "none", facecolors='black')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='black')


plt.errorbar(x_axis, sgd, linestyle='--', lw=4, color="royalblue", label=r"$\mathrm{SGD}$")
plt.fill_between(
    x_axis, 
    sgd + sgd_std, 
    sgd - sgd_std, color="royalblue", alpha=0.2
)

plt.errorbar(x_axis, ours, linestyle='solid', lw=4, color="royalblue", label=r"$\mathrm{NSO}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="royalblue", alpha=0.2
)



ax.set_xlabel(r"$\mathrm{Number~of~Epochs}$", fontsize = 28)
ax.set_ylabel(r"$\mathrm{Test~loss}$", fontsize = 28)
ax.tick_params(labelsize=28)
# ax.set_ylim([-300, 9000]) 
# plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])

ax.set_ylim([0.5, 0.75]) 
plt.xticks(np.arange(0, 7, 1), [r"$0$", r"$5$", r"$10$", r"$15$", r"$20$", r"$25$", r"$30$"])
ax.set_title(r'$\mathrm{ResNet}$'+'-'+r'$\mathrm{34}$', fontsize=28)

# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(28)


plt.legend(fontsize=22)

ax.grid(lw=0.8)
plt.tight_layout()
plt.savefig(f"./resnet_cifar100_test_losses.pdf", format="pdf", dpi=1200)
plt.show()
