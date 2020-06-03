import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
from scipy.optimize import curve_fit
import sys
sys.path.append('../scripts/')
from plot_utils import set_size

data = np.load('../script_output/data_dyn_decoupl.npz')

taus = data["taus"]
proj1 = data["proj1"]
proj2 = data["proj2"]
an_proj1 = data["an_proj1"]
an_proj2 = data["an_proj2"]

def analytic_dd(tau, N, args):
    A = args[1] * np.cos(args[2])
    B = args[1] * np.sin(args[2])
    wL = args[0]
    w_tilde = np.sqrt((A + wL)**2 + B**2)
    mz = (A + wL) / w_tilde
    mx = B / w_tilde
    alpha = w_tilde * tau
    beta = wL * tau
    term1 = np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta)
    term2 = mx**2 * (1 - np.cos(alpha)) * (1 - np.cos(beta)) / (1 + term1)
    M = 1 - term2 * np.power(np.sin(N * np.arccos(term1) / 2), 2)
    return (M + 1) / 2


plt.plot(taus, proj1, label='q1')
plt.plot(taus, proj2, label='q2')
plt.plot(taus, an_proj1, label='an_q1')
plt.plot(taus, an_proj2, label='an_q2')
plt.ylabel(r'$P_x$')
plt.xlabel(r'$\tau$')
plt.tight_layout()
plt.legend()

plt.show()
