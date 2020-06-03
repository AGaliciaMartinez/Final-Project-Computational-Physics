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

    # initial density matrix
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
    return np.trace(rho_last @ np.kron(sx, si))


def analytic(tau, N, args):
    w_tilde = np.sqrt((args[1] * np.cos(args[2] + args[0]))**2 +
                      (args[1] * np.sin(args[2]))**2)
    mz = (args[1] * np.cos(args[2]) + args[0]) / w_tilde
    mx = args[1] * np.sin(args[2]) / w_tilde
    phi = np.arccos(
        np.cos(w_tilde * tau) * np.cos(args[0] * tau) -
        mz * np.sin(w_tilde * tau) * np.sin(args[0] * tau))
    term2 = ((np.power(mx, 2)) * (1 - np.cos(w_tilde * tau)) *
             (1 - np.cos(args[0] * tau))) / (
                 1 + np.cos(w_tilde * tau) * np.cos(args[0] * tau) -
                 mz * np.sin(w_tilde * tau) * np.sin(args[0] * tau))
    M = 1 - term2 * (np.sin(N * phi / 2))**2
    Px = (M + 1) / 2

    return Px


if __name__ == '__main__':
    steps = 25
    N = 8
    rho_0 = np.kron((si + sx) / 2, si / 2)
    taus = np.linspace(0.00, 10.00, 10000)

    args1 = [1.0, 0.1, np.pi / 4]
    # parameters1 = zip(repeat(H), repeat(rho_0), repeat(N), taus, repeat(steps),
    #                   repeat(args1[0]), repeat(args1[1]), repeat(args1[2]))
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results1 = list(tqdm(pool.starmap(dynamical_decoupling, parameters1)))

    # args2 = [1.0, 0.11, np.pi / 6]
    # parameters2 = zip(repeat(H), repeat(rho_0), repeat(N), taus, repeat(steps),
    #                   repeat(args2[0]), repeat(args2[1]), repeat(args2[2]))
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results2 = list(tqdm(pool.starmap(dynamical_decoupling, parameters2)))

    # plt.plot(taus, results1)
    # plt.plot(taus, results2)
    # for i, tu in enumerate(taus):
    proj = analytic(taus, N, args1)
    plt.plot(taus, proj)
    plt.show()
