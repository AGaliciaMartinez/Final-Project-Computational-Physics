import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from dynamical_decoupling import dynamical_decoupling
from hamiltonians import single_carbon_H
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
import parmap


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
    steps = 25
    N = 32
    rho_0 = np.kron(init_qubit([1, 0, 0]), init_qubit([0, 0, 0]))
    taus = np.linspace(9.00, 15.00, 10)

    args1 = [1.0, 0.1, np.pi / 4]

    parameters1 = list(
        product([single_carbon_H], [rho_0], [N], taus, [steps], [args1[0]],
                [args1[1]], [args1[2]]))
    results1 = parmap.starmap(dynamical_decoupling,
                              parameters1,
                              pm_pbar=True,
                              pm_chunksize=3)

    args2 = [1.0, 0.2, np.pi / 6]
    parameters2 = list(
        product([single_carbon_H], [rho_0], [N], taus, [steps], [args2[0]],
                [args2[1]], [args2[2]]))
    results2 = parmap.starmap(dynamical_decoupling,
                              parameters2,
                              pm_pbar=True,
                              pm_chunksize=3)

    an_proj1 = analytic_dd(taus, N, args1)
    an_proj2 = analytic_dd(taus, N, args2)

    # np.savez("../script_output/data_dyn_decoupl_single_N_" + str(N) +
    #          "_steps_" + str(steps),
    #          proj1=results1,
    #          proj2=results2,
    #          an_proj1=an_proj1,
    #          an_proj2=an_proj2,
    #          taus=taus)

    plt.plot(taus, results1, label='q1 sim')
    plt.plot(taus, results2, label='q2 sim')

    plt.plot(taus, an_proj1, label='q1 an')
    plt.plot(taus, an_proj2, label='q2 an')

    plt.legend()
    plt.show()
