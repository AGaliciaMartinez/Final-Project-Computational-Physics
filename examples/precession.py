import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
sys.path.append('../')
from scripts.lindblad_solver import lindblad_solver
from scripts.utils import sx, sy, sz, si
from scripts.plot_utils import set_size

def H(t):
    return 2 * np.pi * sz

# initial state density matrix
rho_0 = (si + (sz + sx) / np.sqrt(2)) / 2

tlist = np.linspace(0, 5, 1000)

rho, expect = lindblad_solver(H,
                              rho_0,
                              tlist,
                              c_ops=[],
                              e_ops=[np.eye(2), sx, sy, sz])

# nice_fonts = {
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "serif",
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": 10,
#     "font.size": 10,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
# }

# mpl.rcParams.update(nice_fonts)

# fig, ax = plt.subplots(1,
#                        1,
#                        figsize=set_size(width='report_full'))

# plt.figure(1)
plt.plot(tlist, expect[:, 0], label='I')
plt.plot(tlist, expect[:, 1], label='X')
plt.plot(tlist, expect[:, 2], label='Y')
plt.plot(tlist, expect[:, 3], label='Z')
plt.legend()
# plt.tight_layout()
plt.show()
