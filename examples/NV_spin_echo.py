import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../scripts/')
from lindblad_solver import lindblad_solver
from utils import si, sx, sy, sz, init_qubit
from tqdm import tqdm

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


def H(t):
    return sz * np.pi


tau_list = np.linspace(0, 3, 100)
e = []
for tau in tqdm(tau_list):
    rho = init_qubit([1, 0, 0])
    tlist = np.linspace(0, tau, 100)
    rho, expect = lindblad_solver(H,
                                  rho,
                                  tlist,
                                  c_ops=[np.sqrt(0.5) * sz],
                                  e_ops=[])
    rho = sx @ rho @ sx
    rho, expect = lindblad_solver(H,
                                  rho,
                                  tlist,
                                  c_ops=[np.sqrt(0.5) * sz],
                                  e_ops=[sx])
    e.append(1 / 2 + np.trace(rho @ sx) / 2)

plt.figure(figsize=(6, 3))

plt.plot(2 * tau_list, e, label=r"Simulation with $T_2$ = 1")
# plt.plot(2 * tau_list, 1 / 2 + 1 / 2 * np.exp(-2 * tau_list))
plt.ylabel(r'$P_+$')
plt.xlabel('$t$')
plt.ylim(0.49, 1.1)
plt.tight_layout()
plt.legend()
plt.savefig('../presentation/images/spin_echo_NV_collapse.svg')
plt.show()
