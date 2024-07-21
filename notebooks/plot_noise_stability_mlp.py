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
0.020,
0.021,
0.022,
0.023,
0.024,
0.025,
0.026,
0.027,
0.028,
0.029,
0.030,])

noise_stability = np.array([
0.0122,
0.0124,
0.0137,
0.0142,
0.0152,
0.0175,
0.0182,
0.0209,
0.0215,
0.0244,
0.0258,
])

noise_stability_std = np.array([
0.0037,
0.0036,
0.0042,
0.0049,
0.0046,
0.0047,
0.0048,
0.0045,
0.0049,
0.0075,
0.0059,
])

Hessian_approx = np.array([
0.0096,
0.0106,
0.0117,
0.0128,
0.0139,
0.0151,
0.0163,
0.0176,
0.0189,
0.0203,
0.0218,
])

f, ax = plt.subplots(figsize=(7,5.5)) 


for i in range(len(sigmas)):
    scatter2 = ax.scatter(sigmas[i], noise_stability[i], s=80, marker="o", edgecolors = "none", facecolors='black')

plt.plot(sigmas, noise_stability, linestyle='solid', lw=4, color="black", label=r"$\mathrm{Gap}$")
plt.fill_between(
    sigmas, 
    noise_stability + noise_stability_std*0.7, 
    noise_stability - noise_stability_std*0.7, color="r", alpha=0.3
)

plt.plot(sigmas, Hessian_approx, linestyle='--', lw=4, color="black", label=r"$\mathrm{Trace}$")

ax.set_xlabel(r"$\sigma$", fontsize = 36)
# ax.set_ylabel(r"$\mathrm{Trace}$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([0.004, 0.035]) 
# # plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])
# plt.yticks(np.arange(1000, 12001, 3000))#, [r"$0.1$", r"$0.4$", r"$0.7$", r"$1.0$", r"$1.3$"])

# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
#plt.yticks(np.arange(0, 14001, 3000))
# plt.xticks(np.arange(0, 7, 2))

# plt.gca().invert_xaxis()
ax.set_title(r'$\mathrm{MLP}$', fontsize=32) # + r' $(\times$' + r'$10^4)$'

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)


plt.legend(fontsize=30)

ax.grid(lw=0.5,ls=":")
plt.tight_layout()
plt.savefig(f"./plot_noise_stability_mlp_mnist.pdf", format="pdf", dpi=1200)
plt.show()
