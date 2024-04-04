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

sgd = np.array([1.1     , 0.90213 , 0.805068, 0.74586 , 0.680566, 0.656127,
       0.6558  ])
sgd_std = np.array([0.00909092, 0.00639784, 0.00760075, 0.0084096 , 0.00747858,
       0.0080027 , 0.00559905])

ours = np.array([1.1     , 0.834862, 0.737149, 0.678275, 0.630311, 0.605196,
       0.601454])
ours_std = np.array([0.00909092, 0.00835498, 0.00827346, 0.00909812, 0.01096873,
       0.00839749, 0.00657807])


f, ax = plt.subplots(figsize=(6,5.5)) 

x_axis = np.arange(len(sgd))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], sgd[i], s=80, marker="o", edgecolors = "none", facecolors='black')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], ours[i], s=80, marker="o", edgecolors = "none", facecolors='black')


plt.errorbar(x_axis, sgd, linestyle='--', lw=6, color="royalblue", label=r"$\mathrm{SGD}$")
plt.fill_between(
    x_axis, 
    sgd + sgd_std, 
    sgd - sgd_std, color="royalblue", alpha=0.1
)

plt.errorbar(x_axis, ours, linestyle='solid', lw=6, color="royalblue", label=r"$\mathrm{NSO}$")
plt.fill_between(
    x_axis, 
    ours + ours_std, 
    ours - ours_std, color="royalblue", alpha=0.1
)

ax.set_xlabel(r"$t$", fontsize = 36)
ax.set_ylabel(r"$L(f_W)$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([0.57, 1.03])
plt.yticks(np.arange(0.6, 1.1, 0.2))
plt.xticks(np.arange(0, 7, 2))

# plt.gca().invert_xaxis()
# ax.set_title(r'$\mathrm{BERT}$'+'-'+r'$\mathrm{Base}$', fontsize=42)
ax.set_title(r'$\mathrm{BERT}$', fontsize=32)


plt.legend(fontsize=32, loc="upper right")

ax.grid(lw=0.8)
plt.tight_layout()
plt.savefig(f"./bert_mrpc_test_losses.pdf", format="pdf", dpi=1200)
plt.show()
