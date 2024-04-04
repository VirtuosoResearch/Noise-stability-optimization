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
8282.688062, 10816.91605, 7462.843081, 7004.905417, 6338.32564, 6177.502564, 6083.152201
])
sgd_std = np.array([ 32.72729738, 559.99459686, 490.0188606 , 510.2399913 ,
       286.96455482, 290.0674835 , 239.97624494])

ours = np.array([
8282.688062, 7622.003471, 5186.342274,  1705.897196,  1498.425938, 1295.754717, 1189.21709
])

ours_std = np.array([ 32.72729738, 208.87454871, 206.83659249, 227.45298708,
       274.21835409, 209.93731929, 164.84517783])


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

ax.set_xlabel(r"$\mathrm{Number~of~Epochs}$", fontsize = 36)
#ax.set_ylabel(r"$\mathrm{Trace}$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([-1000, 14000]) 
# plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])
plt.yticks(np.arange(1000, 14001, 3000), [r"$0.1$", r"$0.4$", r"$0.7$", r"$1.0$", r"$1.3$"])

# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
#plt.yticks(np.arange(0, 14001, 3000))
plt.xticks(np.arange(0, 7, 1))

# plt.gca().invert_xaxis()
ax.set_title(r'$\mathrm{Hessian~Trace}$' + r' $(\times$' + r'$10^4)$', fontsize=36)

#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.get_offset_text().set_fontsize(28)


#plt.legend(fontsize=36)

ax.grid(lw=0.2)
plt.tight_layout()
plt.savefig(f"./bert_mrpc_hessian_traces.pdf", format="pdf", dpi=1200)
plt.show()


# %%
