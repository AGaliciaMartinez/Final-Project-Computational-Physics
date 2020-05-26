import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from scripts.lindblad_solver import lindblad_solver
from scripts.utils import sx, sy, sz, si

# def rotation(t):
#     freq = 1 / 2
#     if t <= 1:
#         Ham = freq * 2 * np.pi * sz
#     elif t > 1:
#         if t % 1 <= 0.5:
#             Ham = 0.01 * 2 * np.pi * sx + 1 / 2 * 2 * np.pi * sz
#         elif t % 1 > 0.5:
#             Ham = freq * 2 * np.pi * sz
#     return Ham

# # initial state density matrix
# # rho_0 = (si + sx) / 2
# rho_0 = (si + sz) / 2

# tlist = np.linspace(0, 200, 10000)

# rho, expect = lindblad_solver(rotation,
#                               rho_0,
#                               tlist,
#                               c_ops=[],
#                               e_ops=[np.eye(2), sx, sy, sz])

# plt.plot(tlist, expect[:, 0], label='I')
# plt.plot(tlist, expect[:, 1], label='X')
# plt.plot(tlist, expect[:, 2], label='Y')
# plt.plot(tlist, expect[:, 3], label='Z')
# plt.legend()
# plt.show()

# import qutip
# # generate test matrix (using qutip for convenience)
# dm = qutip.rand_dm_hs(8, dims=[[2, 4]] * 2).full()
# print(dm)
# # reshape to do the partial trace easily using np.einsum
# reshaped_dm = dm.reshape([2, 4, 2, 4])
# # compute the partial trace
# reduced_dm = np.einsum('ijik->jk', reshaped_dm)

dm = 1 / 2 * np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0],
                       [0, 0, 0, 0]])
print(dm)
reshaped_dm = dm.reshape([2, 2, 2, 2])
print(reshaped_dm)
reduced_dm1 = np.einsum('jiki->jk', reshaped_dm)
reduced_dm2 = np.einsum('ijik->jk', reshaped_dm)
print(reduced_dm1)
print(reduced_dm2)
