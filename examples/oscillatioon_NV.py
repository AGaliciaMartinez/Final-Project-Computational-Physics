import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../scripts/')
from lindblad_solver import lindblad_solver
from utils import si, sx, sy, sz, init_qubit

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


def H(t, frequency):
    return sz * frequency / 2


rho_0 = init_qubit([1, 0, 0])
tlist = np.linspace(0, 5, 1000)

frequency = 1 * 2 * np.pi
rho, expect = lindblad_solver(
    H,
    rho_0,
    tlist,
    frequency,  # Extra argument to H
    # c_ops=[np.sqrt(0.05) * sz],
    e_ops=[si, sx, sy, sz])

plt.figure(figsize=(6, 3))

# plt.plot(tlist, expect[0, :], label='I')
plt.plot(tlist, expect[1, :], label='X')
# plt.plot(tlist, expect[2, :], label='Y')
# plt.plot(tlist, expect[3, :], label='Z')
plt.ylabel(r'$\langle \sigma_x (t)\rangle$')
plt.xlabel('$t$')
plt.tight_layout()
plt.savefig('../presentation/images/spin_precesion.svg')
plt.show()
