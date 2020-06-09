import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
from scipy.optimize import curve_fit
import sys
sys.path.append('../scripts/')
from plot_utils import set_size

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

mpl.rcParams.update(nice_fonts)


def fidelity(t, A, T_2, n):
    return 1 / 2 + A / 2 * np.exp(-(t / T_2)**n)


N = [1]

plt.figure(figsize=(6, 3))
for i in N:
    time = np.load('../script_output/nv_deco_time.npz')['time']
    data = np.load(f'../script_output/nv_deco_{i}.npz')['result']
    mean = data[0, 0]
    std = data[1, 0]

    fit_mean, fit_std = curve_fit(fidelity, time, mean, p0=(1, 300, 3))

    plt.errorbar(time, mean, std, label=f'N={i}')
    A, T_2, n = fit_mean
    plt.plot(time, fidelity(time, A, T_2, n))

    print(f'N={i}')
    print(f'A={A} pm {fit_std[0,0]}')
    print(f'T_2={T_2} pm {fit_std[1,1]}')
    print(f'n={n} pm {fit_std[2,2]}')

plt.title('Dynamical Decoupling on NV-centre')
plt.ylabel(r'$P_x$')
plt.xlabel(r'$\tau$')
plt.xscale('log')
plt.tight_layout()
plt.legend()
plt.savefig('../presentation/images/nv_decoherence.svg')
plt.show()
