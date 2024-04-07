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
3826.158757, 7382.730301, 5371.683095, 4905.58065, 4758.425716, 4657.543113, 4778.289948
])
#0.599731052, 0.9064214797,
sgd_std = np.array([ 16.36364869, 279.99729843, 245.0094303 , 255.11999565,
       143.48227741, 145.03374175, 119.98812247])

ours = np.array([
3826.158757, 3580.25212, 3223.857877, 2461.083468, 2325.587881, 2275.3467, 2138.865354
])

ours_std = np.array([16.36364869, 69.62484957, 68.94553083, 75.81766236, 91.40611803,
       69.97910643, 54.94839261])


# %%
f, ax = plt.subplots(figsize=(6,5.5)) 

x_axis = np.arange(len(sgd))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], sgd[i], s=80, marker="o", edgecolors = "none", facecolors='black')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='black')

# ax.plot(x_axis, auc_clintox,  lw=3, color="darkblue", ls='solid')
# ax.plot(x_axis, auc_bbbp,  lw=3, color="darkred", ls='solid')

plt.errorbar(x_axis, sgd, linestyle='--', lw=4, color="royalblue", label=r"$\mathrm{SGD}$")
plt.fill_between(
    x_axis, 
    sgd + sgd_std, 
    sgd - sgd_std, color="royalblue", alpha=0.3
)

plt.errorbar(x_axis, ours, linestyle='solid', lw=4, color="royalblue", label=r"$\mathrm{Alg.~1}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="royalblue", alpha=0.3
)

ax.set_xlabel(r"$t$", fontsize = 36)
ax.set_ylabel(r"$\nabla^2 \hat L(f_W)$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([1500, 8000]) 
#plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])

# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
plt.yticks(np.arange(2000, 8001, 2000))#, [r"$1.0$", r"$3.0$", r"$5.0$", r"$7.0$"])
#plt.yticks([])

plt.xticks(np.arange(0, 7, 2), [r"$0$", r"$10$", r"$20$", r"$30$"])
ax.set_title(r'$\mathrm{ResNet}$', fontsize=32)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

# plt.gca().invert_xaxis()

#plt.legend(fontsize=36)

ax.grid(lw=0.8)
plt.tight_layout()
plt.savefig(f"./resnet_cifar100_hessian_traces.pdf", format="pdf", dpi=1200)
plt.show()


# %%
