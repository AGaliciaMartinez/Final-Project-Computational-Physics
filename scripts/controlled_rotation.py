import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from tqdm import tqdm
from itertools import repeat
import multiprocessing as mp


def H(t, wL, wh, theta):
    """
    Definition of the Hamiltonian for a single Carbon near a
    Nitrogen-Vacancy centre in diamond.

    Input:
    wL - the Larmor frequency of precession, controlled by the
    externally applied B field

    wh - the hyperfine coupling term describing the strength of
    spin-spin interaction between the Carbon and the NV

    theta - the angle between the applied B field and the vector
    pointing from the NV to the Carbon atom

    Output:
    The 4x4 Hamiltonian of the joint spin system.
    """
    A = wh * np.cos(theta)
    B = wh * np.sin(theta)
    fac = 2 * np.pi
    return fac * (A * np.kron((si - sz) / 2, sz) + B * np.kron(
        (si - sz) / 2, sx) + wL * np.kron((si + sz) / 2, sz))


def dynamical_decoupling(H, rho_0, N, tau, steps, *args):
    """
    Input:
    tau - the free evolution time in the dynamical decoupling sequence
    described by tau - R(pi) - 2tau - R(pi) - tau pulses

    Output:
    Returns the projection along the x axis of the eletron's state after
    N decoupling sequences.
    """
    time1 = np.linspace(0, tau, steps)
    time2 = np.linspace(tau, 3 * tau, 2 * steps)
    time3 = np.linspace(3 * tau, 4 * tau, steps)

    # initial density matrix for ms=0
    rho_last = rho_0

    # implement N dynamical decoupling cycles
    for i in range(N):
        # tau evolution
        rho, _ = lindblad_solver(H, rho_last, time1, *args, c_ops=[], e_ops=[])
        rho_down = np.kron(sx, si) @ rho @ np.kron(sx, si)
        # 2 tau flipped evolution
        rho_fl, _ = lindblad_solver(H,
                                    rho_down,
                                    time2,
                                    *args,
                                    c_ops=[],
                                    e_ops=[])
        rho_up = np.kron(sx, si) @ rho_fl @ np.kron(sx, si)
        # tau flipped back evolution
        rho_flb, _ = lindblad_solver(H,
                                     rho_up,
                                     time3,
                                     *args,
                                     c_ops=[],
                                     e_ops=[])

        rho_last = rho_flb

        time1 = time1 + 4 * tau
        time2 = time2 + 4 * tau
        time3 = time3 + 4 * tau

    exp = [
        np.trace(rho_last @ np.kron(si, sx)),
        np.trace(rho_last @ np.kron(si, sy)),
        np.trace(rho_last @ np.kron(si, sz))
    ]
    return exp[0], exp[1], exp[2]


if __name__ == "__main__":

    steps = 200
    args = [1.0, 0.01, np.pi / 4]
    rho_0 = np.kron((si - sz) / 2, (si + sz) / 2)
    tau = 1.2422
    N = np.arange(0, 33)
    case = 0  # 0 for e- in |0> state, 1 for |1>

    if case == 0:
        rho_0 = np.kron((si + sz) / 2, (si + sz) / 2)
    elif case == 1:
        rho_0 = np.kron((si - sz) / 2, (si + sz) / 2)

    parameters = zip(repeat(H), repeat(rho_0), N, repeat(tau), repeat(steps),
                     repeat(args[0]), repeat(args[1]), repeat(args[2]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(dynamical_decoupling, parameters)

    px = np.zeros(len(N))
    py = np.zeros(len(N))
    pz = np.zeros(len(N))
    for i in N - 1:
        px[i] = results[i][0]
        py[i] = results[i][1]
        pz[i] = results[i][2]

    plt.plot(N, px, label='X', color='green')
    plt.plot(N, py, label='Y', color='red')
    plt.plot(N, pz, label='Z', color='black')
    plt.legend()
    plt.show()

    # rho_0 = np.kron((si + sx) / 2, si / 2)
    # N = 8
    # taus = np.linspace(1.00, 2.00, 100)
    # parameters = zip(repeat(H), repeat(rho_0), repeat(N), taus, repeat(steps),
    #                  repeat(args[0]), repeat(args[1]), repeat(args[2]))
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.starmap(dynamical_decoupling, parameters)

    # plt.plot(taus, results)
    # plt.show()
