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

case = 'pair'

if case == 'single':
    data = np.load('../script_output/data_dyn_decoupl_single_N_32_steps_50.npz')
    taus = data["taus"]
    proj1 = data["proj1"]
    proj2 = data["proj2"]
    an_proj1 = data["an_proj1"]
    an_proj2 = data["an_proj2"]


    fig, ax = plt.subplots(1,
                           1,
                           figsize=set_size(width='report_full'))

    plt.figure(1)
    plt.plot(taus, proj1)
    plt.plot(taus, proj2)
    plt.plot(taus, an_proj1, label='Carbon 1', color='blue')
    plt.plot(taus, an_proj2, label='Carbon 2', color='green')
    plt.title('Dynamical Decoupling on 2 Isolated Carbon-13 Atoms')
    plt.ylabel(r'$P_x$')
    plt.xlabel(r'$\tau$')
    plt.tight_layout()
    plt.legend()
    plt.show()


def interaction_times(k, X, Z):
    wR = np.sqrt(X**2 + (Z / 2)**2)
    return (2 * k - 1) * np.pi / (2 * wR)

if case == 'pair':
    data = np.load('../script_output/data_dyn_decoupl_pair_N_8_steps_200.npz')
    taus = data["taus"]
    proj1 = data["proj1"]
    proj2 = data["proj2"]
    args1 = data["args1"]
    args2 = data["args2"]
    ks = data["ks"]

    fig, ax = plt.subplots(1,
                           1,
                           figsize=set_size(width='report_full'))

    plt.figure(1)
    plt.plot(taus, proj1, label='q1')
    plt.plot(taus, proj2, label='q2')

    # for i in ks:
    #     plt.axvline(interaction_times(i, args1[0], args1[1]), color='green')
    #     plt.axvline(interaction_times(i, args2[0], args2[1]), color='red')

    plt.ylabel(r'$P_x$')
    plt.xlabel(r'$\tau$')
    plt.tight_layout()
    plt.legend()
    plt.show()
