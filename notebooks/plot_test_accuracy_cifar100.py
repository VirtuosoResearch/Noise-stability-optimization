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

sgd = np.array([0.7     , 0.628499 , 0.6733222, 0.7296892, 0.7616424, 0.7726626,
       0.7807484, 0.7871224, 0.7925706, 0.7980944, 0.8058588, 0.8110634,
       0.8161394, 0.8206546, 0.8224722, 0.8235958, 0.8243196, 0.8242436,
       0.8249052, 0.8250246, 0.8247488, 0.824581 , 0.8252176, 0.8255004,
       0.8255534, 0.8254568, 0.8257302, 0.8256498, 0.825914 , 0.825627 ,
       0.827695 ])
sgd = sgd[[0, 5, 10, 15, 20, 25, 30]]
sgd_std = np.array([0.002312, 0.001712, 0.002112, 0.001912, 0.001924, 0.001541, 0.00212])

ours = np.array([0.7      , 0.642038 , 0.681206 , 0.7387082, 0.7699076, 0.7866814,
       0.7946478, 0.7996686, 0.805073 , 0.8119112, 0.8185244, 0.8253834,
       0.8322054, 0.8371962, 0.839046 , 0.8394296, 0.8403188, 0.8422434,
       0.8421032, 0.842117 , 0.8424502, 0.8423216, 0.8408176, 0.8418148,
       0.8421088, 0.8422812, 0.8427154, 0.84326  , 0.8436462, 0.843589 ,
       0.845703 ])
ours = ours[[0, 5, 10, 15, 20, 25, 30]]
ours_std = np.array([0.001958, 0.001716, 0.001728, 0.001532, 0.001632, 0.001552, 0.00172])


f, ax = plt.subplots(figsize=(6,5)) 

x_axis = np.arange(len(sgd))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], sgd[i], s=80, marker="o", edgecolors = "none", facecolors='black')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='black')

plt.errorbar(x_axis, ours, linestyle='solid', lw=4, color="royalblue", label=r"$\mathrm{NSO}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="royalblue", alpha=0.2
)


plt.errorbar(x_axis, sgd, linestyle='--', lw=4, color="royalblue", label=r"$\mathrm{SGD}$")
plt.fill_between(
    x_axis, 
    sgd + sgd_std, 
    sgd - sgd_std, color="royalblue", alpha=0.2
)

ax.set_xlabel(r"$\mathrm{Number~of~Epochs}$", fontsize = 28)
ax.set_ylabel(r"$\mathrm{Test~accuracy}$", fontsize = 28)
ax.tick_params(labelsize=28)
# ax.set_ylim([-300, 9000]) 
# plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])

ax.set_ylim([0.73, 0.86]) 
plt.xticks(np.arange(0, 7, 1), [r"$0$", r"$5$", r"$10$", r"$15$", r"$20$", r"$25$", r"$30$"])
ax.set_title(r'$\mathrm{ResNet}$'+'-'+r'$\mathrm{34}$', fontsize=28)

# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(28)


plt.legend(fontsize=22)

ax.grid(lw=0.8)
plt.tight_layout()
plt.savefig(f"./resnet_cifar100_test_accuracy.pdf", format="pdf", dpi=1200)
plt.show()
