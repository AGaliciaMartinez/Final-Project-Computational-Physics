import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import multiprocessing as mp
import parmap

import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from hamiltonians import pair_carbons_H
from dynamical_decoupling import dynamical_decoupling

if __name__ == '__main__':
    steps = 400
    N = 8
    rho_0 = np.kron(init_qubit([1, 0, 0]), init_qubit([1, 0, 0]))
    taus = np.linspace(40.00, 50.00, 300)

    args1 = [1.0, 0.15]  # [X, Z]
    parameters1 = list(
        product([pair_carbons_H], [rho_0], [N], taus, [steps], [args1[0]],
                [args1[1]]))
    results1 = parmap.starmap(dynamical_decoupling,
                              parameters1,
                              pm_pbar=True,
                              pm_chunksize=3)

    args2 = [1.0, 0.3]  # [X, Z]
    parameters2 = list(
        product([pair_carbons_H], [rho_0], [N], taus, [steps], [args2[0]],
                [args2[1]]))
    results2 = parmap.starmap(dynamical_decoupling,
                              parameters2,
                              pm_pbar=True,
                              pm_chunksize=3)

    ks = [7, 8]

    np.savez("../script_output/data_dyn_decoupl_pair_N_" + str(N) + "_steps_" +
             str(steps),
             args1=args1,
             args2=args2,
             proj1=results1,
             proj2=results2,
             taus=taus,
             ks=ks)

    plt.plot(taus, results1)
    plt.plot(taus, results2)
    plt.show()
