import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from tqdm import tqdm
from itertools import repeat, product
import multiprocessing as mp

import parmap


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


def dynamical_decoupling(H, rho_0, N, tau, steps, *args, e_ops=[]):
    """Input:
    H - the Hamiltonian describing the NV and carbon interaction

    rho_0 - the initial density matrix of the system

    N - the number of dynamical decoupling units to apply

    tau - the free evolution time in the dynamical decoupling sequence
    described by tau - R(pi) - tau pulses. The rotation is around the X axis
    for the qubit 1.

    steps - the number of steps in each tau evolution

    *args - additional arguments passed to the Hamiltonian

    Output:
    Returns the projection along the x axis of the eletron's state after
    N decoupling sequences.
    """
    time = np.linspace(0, tau, steps)
    time_total = np.linspace(0, 2 * N * tau, 2 * N * (steps - 1) - 1)

    # Create the x rotation
    rot = sx
    # TODO a more thorough check would be nice.
    n_qubits = int(np.log2(rho_0.shape[0]))
    for i in range(n_qubits - 1):
        rot = np.kron(rot, si)

    # initial density matrix
    rho = rho_0
    e_total = []

    # implement N/2 dynamical decoupling units
    for i in range(N):
        # tau evolution
        rho, e = lindblad_solver(H, rho, time, *args, c_ops=[], e_ops=e_ops)
        if len(e_ops):
            e_total.append(e[0:-1])

        # Pi rotation
        rho = rot @ rho @ rot

        # tau flipped evolution
        rho, e = lindblad_solver(H, rho, time, *args, c_ops=[], e_ops=e_ops)
        if len(e_ops):
            e_total.append(e[0:-1])

    if len(e_ops):
        # Careful, last value not being appended.
        e_total = np.array(e_total)
        print(e_total.shape)
        print(time.shape)

        e_total = np.moveaxis(e_total, 0, -1)
        e_total.reshape((len(e_ops), -1))

        return time_total, e_total
    else:
        return (np.trace(rho @ np.kron(sx, si)) + 1) / 2


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
    N = 16
    rho_0 = np.kron(init_qubit([1, 0, 0]), init_qubit([0, 0, 0]))
    taus = np.linspace(9.00, 15.00, 300)

    args1 = [1.0, 0.1, np.pi / 4]
    parameters1 = list(
        product([H], [rho_0], [N], taus, [steps], [args1[0]], [args1[1]],
                [args1[2]]))
    print(len(parameters1))
    results1 = parmap.starmap(dynamical_decoupling, parameters1, pm_pbar=True)

    args2 = [1.0, 0.2, np.pi / 6]
    parameters2 = list(
        product([H], [rho_0], [N], taus, [steps], [args2[0]], [args2[1]],
                [args2[2]]))
    results2 = parmap.starmap(dynamical_decoupling, parameters2, pm_pbar=True)

    proj1 = np.array(results1)
    proj2 = np.array(results2)

    an_proj1 = analytic_dd(taus, N, args1)
    an_proj2 = analytic_dd(taus, N, args2)

    # np.savez("../script_output/data_dyn_decoupl",
    #          proj1=proj1,
    #          proj2=proj2,
    #          an_proj1=an_proj1,
    #          an_proj2=an_proj2,
    #          taus=taus)

    # plt.plot(taus, results1, label='q1 sim')
    # plt.plot(taus, results2, label='q2 sim')

    # proj = analytic_dd(taus, N, args1)
    # plt.plot(taus, proj, label='q1 an')

    # proj2 = analytic_dd(taus, N, args2)
    # plt.plot(taus, proj2, label='q2 an')

    # plt.legend()
    # plt.show()
