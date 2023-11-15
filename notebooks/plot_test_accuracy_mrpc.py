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

sgd = np.array([0.79     , 0.82598 , 0.82898 , 0.833235, 0.8404  , 0.842686,
       0.843588])
sgd_std = np.array([ 0.002272729738, 0.001599459686, 0.001900188606 , 0.002102399913 ,
       0.0018696455482, 0.002000674835 , 0.0013997624494])

ours = np.array([0.79    , 0.838686, 0.840686, 0.843137, 0.85049 , 0.852941,
       0.854392])

ours_std = np.array([ 0.002272729738, 0.0020887454871, 0.0020683659249, 0.0022745298708,
       0.0027421835409, 0.0020993731929, 0.001644517783])


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
ax.set_ylim([0.795, 0.865]) 

plt.xticks(np.arange(0, 7, 1))

# plt.gca().invert_xaxis()
ax.set_title(r'$\mathrm{BERT}$'+'-'+r'$\mathrm{Base}$', fontsize=30)


plt.legend(fontsize=22, loc="lower right")

ax.grid(lw=0.2)
plt.tight_layout()
plt.savefig(f"./bert_mrpc_test_accuracy.pdf", format="pdf", dpi=1200)
plt.show()
