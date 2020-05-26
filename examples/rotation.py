import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
sys.path.append('../')
from scripts.lindblad_solver import lindblad_solver
from scripts.utils import sx, sy, sz, si
from scripts.plot_utils import set_size


def rotation(t):
    freq = 1 / 2
    if t <= 1:
        Ham = freq * 2 * np.pi * sz
    elif t > 1:
        if t % 1 <= 0.5:
            Ham = 0.01 * 2 * np.pi * sx + 1 / 2 * 2 * np.pi * sz
        elif t % 1 > 0.5:
            Ham = freq * 2 * np.pi * sz
    return Ham


# defining the initial density matrix
rho_0 = (si + sz) / 2

tlist = np.linspace(0, 80, 10000)

rho, expect = lindblad_solver(rotation,
                              rho_0,
                              tlist,
                              c_ops=[],
                              e_ops=[np.eye(2), sx, sy, sz])

plt.plot(tlist, expect[:, 0], label='I')
plt.plot(tlist, expect[:, 1], label='X')
plt.plot(tlist, expect[:, 2], label='Y')
plt.plot(tlist, expect[:, 3], label='Z')
plt.legend()
plt.show()
