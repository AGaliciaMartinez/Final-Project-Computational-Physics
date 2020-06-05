import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
from scipy.optimize import curve_fit
import sys
sys.path.append('../scripts/')
from plot_utils import set_size

# data = np.load('../script_output/data_dyn_decoupl_single_N_32_steps_50.npz')
data = np.load('../script_output/data_dyn_decoupl_pair_N_32_steps_25.npz')

case = 'pair'
if case == 'single':
    taus = data["taus"]
    proj1 = data["proj1"]
    proj2 = data["proj2"]
    an_proj1 = data["an_proj1"]
    an_proj2 = data["an_proj2"]

    plt.plot(taus, proj1, label='q1')
    plt.plot(taus, proj2, label='q2')
    plt.plot(taus, an_proj1, label='an_q1')
    plt.plot(taus, an_proj2, label='an_q2')
    plt.ylabel(r'$P_x$')
    plt.xlabel(r'$\tau$')
    plt.tight_layout()
    plt.legend()
    plt.show()



def interaction_times(k, X, Z):
    wR = np.sqrt(X**2 + (Z / 2)**2)
    return (2 * k - 1) * np.pi / (2 * wR)

if case == 'pair':
    taus = data["taus"]
    proj1 = data["proj1"]
    proj2 = data["proj2"]
    args1 = data["args1"]
    args2 = data["args2"]
    ks = data["ks"]

    plt.plot(taus, proj1, label='q1')
    plt.plot(taus, proj2, label='q2')

    for i in ks:
        plt.axvline(interaction_times(i, args1[0], args1[1]), color='green')
        plt.axvline(interaction_times(i, args2[0], args2[1]), color='red')

    plt.ylabel(r'$P_x$')
    plt.xlabel(r'$\tau$')
    plt.tight_layout()
    plt.legend()
    plt.show()
