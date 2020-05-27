import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si
from lindblad_solver import lindblad_solver

wL = 1.0
wh = 0.01
theta = np.pi / 4
A = wh * np.cos(theta)
B = wh * np.sin(theta)
fac = 2 * np.pi

rho_0 = 1 / 2 * np.kron((si + sz) / np.sqrt(2), (si + sz) / np.sqrt(2))


def H(t):
    return fac * (A * np.kron(sz, sz) + B * np.kron(sz, sx) +
                  wL * np.kron(si, sz))


print(H(0))


def dynamical_decoupling(H,
                         rho_0,
                         tau,
                         dt,
                         N,
                         c_ops=[],
                         e_ops=[],
                         debug=False):
    time1 = np.arange(0, tau, dt)
    time2 = np.arange(0, 2 * tau, dt)
    rho_last = rho_0
    expectations = np.empty((0, len(e_ops)))
    for i in range(N):
        # tau evolution
        rho, expect = lindblad_solver(H,
                                      rho_last,
                                      time1,
                                      c_ops=[],
                                      e_ops=e_ops)
        rho_next = np.kron(sx, si) @ rho @ np.kron(sx, si)
        # 2 tau flipped evolution
        rho_fl, expect_fl = lindblad_solver(H,
                                            rho_next,
                                            time2,
                                            c_ops=[],
                                            e_ops=e_ops)
        rho_next2 = np.kron(sx, si) @ rho_fl @ np.kron(sx, si)
        # tau flipped back evolution
        rho_flb, expect_flb = lindblad_solver(H,
                                              rho_next2,
                                              time1,
                                              c_ops=[],
                                              e_ops=e_ops)

        rho_flb = rho_last
        expectations = np.concatenate(
            (expectations, expect, expect_fl, expect_flb), axis=0)

    if debug:
        return rho_last, expectations
    elif not debug:
        return expectations


if __name__ == '__main__':
    tau = 0.5
    dt = 0.01
    N = 16
    tlist = np.linspace(0, 4 * tau * N, int(4 * tau * N / dt))

    e_ops = [
        np.kron(si, si),
        np.kron(sx, si),
        np.kron(sy, si),
        np.kron(sz, si),
        np.kron(si, sx),
        np.kron(si, sy),
        np.kron(si, sz)
    ]

    rho, expectations = dynamical_decoupling(H,
                                             rho_0,
                                             tau,
                                             dt,
                                             N,
                                             c_ops=[],
                                             e_ops=e_ops,
                                             debug=True)
    # print(expectations)

    # tlist = np.linspace(0, 2, 20)
    plt.plot(tlist, expectations[:, 0], label='I')
    plt.plot(tlist, expectations[:, 1], label='X')
    plt.plot(tlist, expectations[:, 2], label='Y')
    plt.plot(tlist, expectations[:, 3], label='Z')
    plt.legend()
    plt.show()

    plt.plot(tlist, expectations[:, 0], label='I')
    plt.plot(tlist, expectations[:, 4], label='X')
    plt.plot(tlist, expectations[:, 5], label='Y')
    plt.plot(tlist, expectations[:, 6], label='Z')
    plt.legend()
    plt.show()
