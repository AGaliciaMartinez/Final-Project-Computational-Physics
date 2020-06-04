import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from tqdm import tqdm
from itertools import repeat
import multiprocessing as mp


def H(t, X, Z):
    """
    Definition of the Hamiltonian for a Carbon pair near a
    Nitrogen-Vacancy centre in diamond.

    Input:
    X - the strength of the dipolar coupling between the Carbon
    atoms in a pair

    Z - the strength of the hyperfine coupling between the pair
    and the NV centre

    Output:
    The 4x4 Hamiltonian of the joint spin system.
    """
    return X * np.kron((si + sz) / 2, sx / 2) + X * np.kron(
        (si - sz) / 2, sx / 2) + Z * np.kron((si - sz) / 2, sz / 2)


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

    # TODO a more thorough check would be nice.
    n_qubits = np.log2(rho_0.shape[0])

    # Create the x rotation
    rot = sx
    for i in range(int(n_qubits) - 1):
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

        # 2 tau flipped evolution
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


def interaction_times(k, X, Z):
    wR = np.sqrt(X**2 + (Z / 2)**2)
    return (2 * k - 1) * np.pi / (2 * wR)


if __name__ == '__main__':
    steps = 25
    N = 32
    rho_0 = np.kron(init_qubit([1, 0, 0]), init_qubit([1, 0, 0]))
    taus = np.linspace(0.00, 12.00, 300)

    args1 = [1.0, 0.1]  # [X, Z]
    parameters1 = zip(repeat(H), repeat(rho_0), repeat(N), taus, repeat(steps),
                      repeat(args1[0]), repeat(args1[1]))
    with mp.Pool() as pool:
        results1 = list(tqdm(pool.starmap(dynamical_decoupling, parameters1)))

    ks = [1, 2, 3, 4]
    for i in ks:
        plt.axvline(interaction_times(i, args1[0], args1[1]), color='green')
    plt.plot(taus, results1)
    plt.show()
