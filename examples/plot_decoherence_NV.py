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
    "axes.labelsize": 14,
    "font.size": 16,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}

mpl.rcParams.update(nice_fonts)


def fidelity(t, A, T_2, n):
    return 1 / 2 + A / 2 * np.exp(-(t / T_2)**n)


N = [1, 2, 4, 8, 16]
colors = ['purple', 'orange', 'green', 'blue', 'red']

plt.figure(1, figsize=(5, 3))
Ts_mean = []
Ts_std = []
for i, n in enumerate(N):
    time = np.load('../script_output/nv_deco_time.npz')['time']
    data = np.load(f'../script_output/nv_deco_{n}.npz')['result']
    mean = data[0, 0]
    std = data[1, 0]

    fit_mean, fit_std = curve_fit(fidelity, time, mean, p0=(1, 300, 3))
    fit_err = np.sqrt(np.diag(fit_std))

    plt.errorbar(time, mean, std, label=f'N={n}', color=colors[i])
    plt.plot(time, fidelity(time, *fit_mean), '--', color=colors[i])

    A, T_2, n_fit = fit_mean

    Ts_mean.append(T_2)
    Ts_std.append(fit_err[1])
    print(f'N={n}')
    print(f'A={A} pm {fit_err[0]}')
    print(f'T_2={T_2} pm {fit_err[1]}')
    print(f'n={n_fit} pm {fit_err[2]}')
    print('')

plt.title('Dynamical Decoupling on NV-centre')
plt.ylabel(r'$P_x$')
plt.xlabel(r'$t$')
plt.xscale('log')
plt.tight_layout()
plt.legend()
plt.savefig('../presentation/images/nv_decoherence.svg')

N_1 = [1]
colors = ['purple', 'orange', 'green', 'blue', 'red']

plt.figure(2, figsize=(5, 3))
for i, n in enumerate(N_1):
    time = np.load('../script_output/nv_deco_time.npz')['time']
    data = np.load(f'../script_output/nv_deco_{n}.npz')['result']
    mean = data[0, 0]
    std = data[1, 0]

    fit_mean, fit_std = curve_fit(fidelity, time, mean, p0=(1, 300, 3))
    fit_err = np.sqrt(np.diag(fit_std))

    plt.errorbar(time, mean, std, label=f'N={n}', color=colors[i])
    plt.plot(time, fidelity(time, *fit_mean), '--', color=colors[i])

    A, T_2, n_fit = fit_mean

    print(f'N={n}')
    print(f'A={A} pm {fit_err[0]}')
    print(f'T_2={T_2} pm {fit_err[1]}')
    print(f'n={n_fit} pm {fit_err[2]}')
    print('')

plt.title('Dynamical Decoupling on NV-centre')
plt.ylabel(r'$P_x$')
plt.xlabel(r'$t = 2\tau$')
plt.xscale('log')
plt.tight_layout()
plt.legend()
plt.savefig('../presentation/images/nv_decoherence_1.svg')


# Fit of Tau
def f(N, A, exponent):
    return A * N**exponent


fit_mean, fit_std = curve_fit(f, N, Ts_mean, p0=(1, 2 / 3), sigma=Ts_std)
fit_err = np.sqrt(np.diag(fit_std))

print(f'A = {fit_mean[0]} pm {fit_err[0]}')
print(f'exponen = {fit_mean[1]} pm {fit_err[1]}')

plt.figure(3, figsize=(5, 3))
plt.errorbar(N, Ts_mean, Ts_std)
plt.plot(N, f(N, *fit_mean))
plt.title('Scaling of $T_2$ with Dynamical Decoupling')
plt.ylabel(r'$T_2$')
plt.xlabel(r'$N$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('../presentation/images/T_2_NV_1.svg')

np.savez('../script_output/nv_deco_T2.npz', T2_mean=Ts_mean, T2_std=Ts_std)

plt.show()
