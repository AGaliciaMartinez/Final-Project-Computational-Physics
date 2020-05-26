import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si
from lindblad_solver import lindblad_solver

rho_0 = 1 / 2 * np.kron((si + sx) / 2, si)
print(rho_0)

wL = 1.0
wh = 0.01
theta = np.pi / 4
A = wh * np.cos(theta)
B = wh * np.sin(theta)
fac = 2 * np.pi

# H0 = wL * sz
# H1 = H0 + A * sz + B * sx

Ham = fac * (A * np.kron(sz, sz) + B * np.kron(sz, sx) + wL * np.kron(si, sz))
print(Ham)


def H(t):
    return Ham


tlist = np.linspace(0, 100, 1000)
e_ops = [
    np.kron(si, si),
    np.kron(sx, si),
    np.kron(sy, si),
    np.kron(sz, si),
    np.kron(si, sx),
    np.kron(si, sy),
    np.kron(si, sz)
]

rho, expect = lindblad_solver(H, rho_0, tlist, c_ops=[], e_ops=e_ops)

plt.plot(tlist, expect[:, 0], label='I')
plt.plot(tlist, expect[:, 1], label='X')
plt.plot(tlist, expect[:, 2], label='Y')
plt.plot(tlist, expect[:, 3], label='Z')
plt.legend()
plt.show()

plt.plot(tlist, expect[:, 0], label='I')
plt.plot(tlist, expect[:, 4], label='X')
plt.plot(tlist, expect[:, 5], label='Y')
plt.plot(tlist, expect[:, 6], label='Z')
plt.legend()
plt.show()
