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
0.0070,
0.0071,
0.0072,
0.0073,
0.0074,
0.0075,
0.0076,
0.0077,
0.0078,
0.0079,
0.0080,])

noise_stability = np.array([
0.0083,
0.0088,
0.0093,
0.0098,
0.0104,
0.0110,
0.0117,
0.0124,
0.0131,
0.0139,
0.0147,
])

noise_stability_std = np.array([
0.0031,
0.0031,
0.0032,
0.0034,
0.0035,
0.0036,
0.0038,
0.0040,
0.0042,
0.0044,
0.0047,
])

Hessian_approx = np.array([
0.0095,
0.0098,
0.0101,
0.0103,
0.0106,
0.0109,
0.0112,
0.0115,
0.0118,
0.0121,
0.0124,
])

f, ax = plt.subplots(figsize=(7,5.5)) 


for i in range(len(sigmas)):
    scatter2 = ax.scatter(sigmas[i], noise_stability[i], s=80, marker="o", edgecolors = "none", facecolors='black')

plt.plot(sigmas, noise_stability, linestyle='solid', lw=4, color="black", label=r"$\mathrm{Gap}$")
plt.fill_between(
    sigmas, 
    noise_stability + noise_stability_std*0.5, 
    noise_stability - noise_stability_std*0.5, color="r", alpha=0.3
)

plt.plot(sigmas, Hessian_approx, linestyle='--', lw=4, color="black", label=r"$\mathrm{Trace}$")

ax.set_xlabel(r"$\sigma$", fontsize = 36)
# ax.set_ylabel(r"$\mathrm{Trace}$", fontsize = 36)
ax.tick_params(labelsize=36)
ax.set_ylim([0.001, 0.035]) 
# # plt.yticks([3, 6, 9, 12], [r"$10^3$", r"$10^{6}$", r"$10^{9}$", r"$10^{12}$"])
# plt.yticks(np.arange(1000, 12001, 3000))#, [r"$0.1$", r"$0.4$", r"$0.7$", r"$1.0$", r"$1.3$"])

# plt.xticks(np.arange(0.4, 0.81, 0.2), fontsize=28)
# ax.set_xlim([-0.5, 9.5]) 
#plt.yticks(np.arange(0, 14001, 3000))
# plt.xticks(np.arange(0, 7, 2))

# plt.gca().invert_xaxis()
ax.set_title(r'$\mathrm{BERT}$', fontsize=32) # + r' $(\times$' + r'$10^4)$'

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)


plt.legend(fontsize=30)

ax.grid(lw=0.5,ls=":")
plt.tight_layout()
plt.savefig(f"./plot_noise_stability_bert_mrpc.pdf", format="pdf", dpi=1200)
plt.show()