import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from tqdm import tqdm
from itertools import repeat
import multiprocessing as mp


def H1(t):
    wL = 1.0
    wh = 0.01
    theta = np.pi / 4
    A = wh * np.cos(theta)
    B = wh * np.sin(theta)
    fac = 2 * np.pi
    return fac * (A * np.kron((si - sz) / 2, sz) + B * np.kron(
        (si - sz) / 2, sx) + wL * np.kron((si + sz) / 2, sz))


def H2(t):
    wL = 1.0
    wh = 0.012
    theta = np.pi / 4
    A = wh * np.cos(theta)
    B = wh * np.sin(theta)
    fac = 2 * np.pi
    return fac * (A * np.kron((si - sz) / 2, sz) + B * np.kron(
        (si - sz) / 2, sx) + wL * np.kron((si + sz) / 2, sz))


def dynamical_decoupling(H, N, tau, steps):
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
    rho_0 = np.kron((si + sx) / 2, si / 2)
    rho_last = rho_0

    # expectations = np.empty((len(e_ops), 0))
    times = []
    # implement N dynamical decoupling cycles
    for i in range(N):
        # tau evolution
        rho, _ = lindblad_solver(H, rho_last, time1, c_ops=[], e_ops=[])
        rho_down = np.kron(sx, si) @ rho @ np.kron(sx, si)
        # 2 tau flipped evolution
        rho_fl, _ = lindblad_solver(H, rho_down, time2, c_ops=[], e_ops=[])
        rho_up = np.kron(sx, si) @ rho_fl @ np.kron(sx, si)
        # tau flipped back evolution
        rho_flb, _ = lindblad_solver(H, rho_up, time3, c_ops=[], e_ops=[])

        rho_last = rho_flb

        time1 = time1 + 4 * tau
        time2 = time2 + 4 * tau
        time3 = time3 + 4 * tau
    return np.trace(rho_last @ np.kron(sx, si))


if __name__ == '__main__':
    steps = 25
    N = 16
    taus = np.linspace(0.00, 1.00, 300)

    parameters1 = zip(repeat(H1), repeat(N), taus, repeat(steps))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results1 = list(tqdm(pool.starmap(dynamical_decoupling, parameters1)))

    parameters2 = zip(repeat(H2), repeat(N), taus, repeat(steps))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results2 = list(tqdm(pool.starmap(dynamical_decoupling, parameters2)))

    plt.plot(taus, results1)
    plt.plot(taus, results2)
    plt.show()
