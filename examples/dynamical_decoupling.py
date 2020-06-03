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
    return (A + wL) * np.kron((si - sz) / 2, sz / 2) + B * np.kron(
        (si - sz) / 2, sx / 2) + wL * np.kron((si + sz) / 2, sz / 2)


def dynamical_decoupling(H, rho_0, N, tau, steps, *args):
    """
    Input:
    H - the Hamiltonian describing the NV and carbon interaction

    rho_0 - the initial density matrix of the system

    N - two times the number of dynamical decoupling units to apply

    tau - the free evolution time in the dynamical decoupling sequence
    described by tau - R(pi) - 2tau - R(pi) - tau pulses

    steps - the number of steps in each tau evolution

    *args - additional arguments passed to the Hamiltonian

    Output:
    Returns the projection along the x axis of the eletron's state after
    N decoupling sequences.
    """
    time1 = np.linspace(0, tau, steps)
    time2 = np.linspace(tau, 3 * tau, 2 * steps)
    time3 = np.linspace(3 * tau, 4 * tau, steps)

    # initial density matrix
    rho_last = rho_0

    # implement N/2 dynamical decoupling units
    for i in range(int(N / 2)):
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
    return (np.trace(rho_last @ np.kron(sx, si)) + 1) / 2


def analytic_dd(tau, N, args):
    A = args[1] * np.cos(args[2])
    B = args[1] * np.sin(args[2])
    wL = args[0]
    w_tilde = np.sqrt((A + wL)**2 + B**2)
    mz = (A + wL) / w_tilde
    mx = B / w_tilde
    alpha = w_tilde * tau
    beta = wL * tau
    term1 = np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta)
    term2 = mx**2 * (1 - np.cos(alpha)) * (1 - np.cos(beta)) / (1 + term1)
    M = 1 - term2 * np.power(np.sin(N * np.arccos(term1) / 2), 2)
    return (M + 1) / 2


if __name__ == '__main__':
    steps = 50
    N = 32
    rho_0 = np.kron(init_qubit([1, 0, 0]), init_qubit([0, 0, 0]))
    taus = np.linspace(9.00, 15.00, 300)

    args1 = [1.0, 0.1, np.pi / 4]
    parameters1 = zip(repeat(H), repeat(rho_0), repeat(N), taus, repeat(steps),
                      repeat(args1[0]), repeat(args1[1]), repeat(args1[2]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results1 = list(tqdm(pool.starmap(dynamical_decoupling, parameters1)))

    args2 = [1.0, 0.2, np.pi / 6]
    parameters2 = zip(repeat(H), repeat(rho_0), repeat(N), taus, repeat(steps),
                      repeat(args2[0]), repeat(args2[1]), repeat(args2[2]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results2 = list(tqdm(pool.starmap(dynamical_decoupling, parameters2)))

    proj1 = np.array(results1)
    proj2 = np.array(results2)

    an_proj1 = analytic_dd(taus, N, args1)
    an_proj2 = analytic_dd(taus, N, args2)

    np.savez("data_dyn_decoupl",
             proj1=proj1,
             proj2=proj2,
             an_proj1=an_proj1,
             an_proj2=an_proj2,
             taus=taus)

    # plt.plot(taus, results1, label='q1 sim')
    # plt.plot(taus, results2, label='q2 sim')

    # proj = analytic_dd(taus, N, args1)
    # plt.plot(taus, proj, label='q1 an')

    # proj2 = analytic_dd(taus, N, args2)
    # plt.plot(taus, proj2, label='q2 an')

    # plt.legend()
    # plt.show()
