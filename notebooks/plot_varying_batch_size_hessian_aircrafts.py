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
4458, 4186, 3678, 4193
])

nso_std = np.array([82, 81, 46, 64])

sam = np.array([
5591, 4943, 4499, 5034
])

sam_std = np.array([87, 97, 59, 63])


f, ax = plt.subplots(figsize=(7,6)) 

x_axis = np.arange(len(nso))

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], nso[i], s=80, marker="o", edgecolors = "none", facecolors='black')

for i in range(len(x_axis)):
    scatter2 = ax.scatter(x_axis[i], sam[i], s=80, marker="o", edgecolors = "none", facecolors='black')

# ax.plot(x_axis, auc_clintox,  lw=3, color="darkblue", ls='solid')
# ax.plot(x_axis, auc_bbbp,  lw=3, color="darkred", ls='solid')


plt.errorbar(x_axis, sam, linestyle='--', lw=4, color="k", label=r"$\mathrm{SAM}$")
plt.fill_between(
    x_axis, 
    sam + sam_std, 
    sam - sam_std, color="r", alpha=0.3
)
plt.errorbar(x_axis, nso, linestyle='solid', lw=4, color="k", label=r"$\mathrm{NSO}$")
plt.fill_between(
    x_axis, 
    nso + nso_std, 
    nso - nso_std, color="r", alpha=0.3
)

ax.set_xlabel(r"$\mathrm{Batch~size}$", fontsize = 36)
ax.set_ylabel(r"$\mathrm{Trace}$", fontsize = 36)
ax.tick_params(labelsize=36)

plt.yticks(np.arange(3000, 5001, 1000))
ax.set_ylim([3500, 6000]) 


plt.xticks(np.arange(0, 4, 1), [r"$8$", r"$16$", r"$32$", r"$64$"])
#ax.set_title(r'$\mathrm{Hessian~trace}$' + r' $(\times$' + r'$10^3)$', fontsize=36)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(36)

# plt.gca().invert_xaxis()

plt.legend(fontsize=30)

ax.grid(ls=":", lw=0.4)
plt.tight_layout()
plt.savefig(f"./resnet_varying_batch_size_hessian_trace_aircrafts.pdf", format="pdf", dpi=1200)
plt.show()

# %%
