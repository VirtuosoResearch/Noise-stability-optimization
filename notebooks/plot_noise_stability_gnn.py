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

sigmas = np.array([
0.040,
0.041, 
0.042, 
0.043, 
0.044, 
0.045, 
0.046, 
0.047, 
0.048, 
0.049, 
0.050, ])

noise_stability = np.array([
0.0243,
0.0266,
0.0287,
0.0297,
0.0298,
0.0313,
0.0363,
0.0414,
0.0449,
0.0455,
0.0482,
])

noise_stability_std = np.array([
0.0097,
0.0141,
0.0086,
0.0109,
0.0111,
0.0092,
0.0105,
0.0109,
0.0089,
0.0160,
0.0100,
])

Hessian_approx = np.array([
0.0278,
0.0292,
0.0306,
0.0321,
0.0336,
0.0351,
0.0367,
0.0383,
0.0400,
0.0417,
0.0434,
])

f, ax = plt.subplots(figsize=(7,5.5)) 


for i in range(len(sigmas)):
    scatter2 = ax.scatter(sigmas[i], noise_stability[i], s=80, marker="o", edgecolors = "none", facecolors='black')

plt.plot(sigmas, noise_stability, linestyle='solid', lw=4, color="black", label=r"$\mathrm{Gap}$")
plt.fill_between(
    sigmas, 
    noise_stability + noise_stability_std*0.6, 
    noise_stability - noise_stability_std*0.6, color="r", alpha=0.3
)

plt.plot(sigmas, Hessian_approx, linestyle='--', lw=4, color="black", label=r"$\mathrm{Trace}$")

ax.set_xlabel(r"$\sigma$", fontsize = 36)
# ax.set_ylabel(r"$\mathrm{Trace}$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([0.005, 0.075]) 
plt.yticks(np.arange(0.02, 0.07, 0.02))
# plt.yticks(np.arange(1000, 12001, 3000))#, [r"$0.1$", r"$0.4$", r"$0.7$", r"$1.0$", r"$1.3$"])

# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
#plt.yticks(np.arange(0, 14001, 3000))
# plt.xticks(np.arange(0, 7, 2))

# plt.gca().invert_xaxis()
ax.set_title(r'$\mathrm{GNN}$', fontsize=32) # + r' $(\times$' + r'$10^4)$'

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)


plt.legend(fontsize=30)

ax.grid(lw=0.5,ls=":")
plt.tight_layout()
plt.savefig(f"./plot_noise_stability_gcn_collab.pdf", format="pdf", dpi=1200)
plt.show()
