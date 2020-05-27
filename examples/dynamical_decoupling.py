import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from tqdm import tqdm
import multiprocessing as mp

wL = 1.0
wh = 0.01
theta = np.pi / 4
A = wh * np.cos(theta)
B = wh * np.sin(theta)
fac = 2 * np.pi
# rho_0 = np.kron((si + sx) / 2, si / 2)
rho_0 = np.kron((si + sx) / 2, si / 2)


def H(t):
    return fac * (A * np.kron((si - sz) / 2, sz) + B * np.kron(
        (si - sz) / 2, sx) + wL * np.kron((si + sz) / 2, sz))


tau = 0.25
steps = 25
N = 16
# tlist = np.linspace(0, 4 * tau * N, int(4 * tau * N / dt))

e_ops = [np.kron(sx, si)]


def dynamical_decoupling(tau):
    time1 = np.linspace(0, tau, steps)
    time2 = np.linspace(tau, 3 * tau, 2 * steps)
    time3 = np.linspace(3 * tau, 4 * tau, steps)
    rho_last = rho_0
    expectations = np.empty((0, len(e_ops)))
    times = []
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
                                              time3,
                                              c_ops=[],
                                              e_ops=e_ops)

        rho_last = rho_flb
        expectations = np.concatenate(
            (expectations, expect[:-1], expect_fl[:-1], expect_flb[:-1]),
            axis=0)
        times = np.concatenate((times, time1[:-1], time2[:-1], time3[:-1]),
                               axis=0)

        time1 = time1 + 4 * tau
        time2 = time2 + 4 * tau
        time3 = time3 + 4 * tau
    return expectations[-1, 0]


if __name__ == '__main__':

    taus = np.linspace(0.05, 2, 99)
    Px = []

with mp.Pool(processes=mp.cpu_count()
             ) as pool:  # By default maximum number of processes.
    results = list(tqdm(pool.imap(dynamical_decoupling, taus),
                        total=len(taus)))

    # for tau in tqdm(taus):
    #     rho, expectations, times = dynamical_decoupling(H,
    #                                                     rho_0,
    #                                                     tau,
    #                                                     steps,
    #                                                     N,
    #                                                     c_ops=[],
    #                                                     e_ops=e_ops,
    #                                                     debug=True)
    # Px.append(expectations[-1, 1])

    plt.plot(taus, results)
    plt.show()

    # rho, expectations, times = dynamical_decoupling(H,
    #                                                 rho_0,
    #                                                 tau,
    #                                                 steps,
    #                                                 N,
    #                                                 c_ops=[],
    #                                                 e_ops=e_ops,
    #                                                 debug=True)

    # # plt.plot(tlist, expectations[:, 0], label='I')
    # # plt.plot(tlist, expectations[:, 1], label='X')
    # # plt.plot(tlist, expectations[:, 2], label='Y')
    # plt.plot(times, expectations[:, 3], label='Z')
    # # plt.legend()
    # # plt.show()

    # plt.plot(times, expectations[:, 0], label='I')
    # plt.plot(times, expectations[:, 4], label='X')
    # plt.plot(times, expectations[:, 5], label='Y')
    # plt.plot(times, expectations[:, 6], label='Z')
    # plt.legend()
    # plt.show()
