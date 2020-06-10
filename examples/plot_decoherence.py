import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm
import sys
sys.path.append('../scripts/')

from utils import normal_autocorr_generator

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 14,
    "font.size": 16,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
mpl.rcParams.update(nice_fonts)

dt = 1.4 / 5
dw = normal_autocorr_generator(0, 0.05, 100000 / dt, 1)
t = np.arange(0, 1000, dt)

dw_list = [next(dw) for i in t]
plt.figure(figsize=(6, 3))
plt.plot(t, dw_list, label="$\tau_c = 10^5, b= 0.05")
plt.ylim(0, 0.02)
plt.ylabel(r'$\delta \omega$')
plt.xlabel('t')
plt.title('Correlated noise employed in the simulations.')
plt.tight_layout()
plt.savefig('../presentation/images/noise_figure.svg')
# plt.show()
