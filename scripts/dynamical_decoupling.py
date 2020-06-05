import numpy as np
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver


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

    # implement N dynamical decoupling units
    for i in range(N):
        # tau evolution
        rho, e = lindblad_solver(H, rho, time, *args, c_ops=[], e_ops=e_ops)
        if i == 0:
            e_total.append(e[:, 0])

        # Pi rotation
        rho = rot @ rho @ rot

        # tau flipped evolution
        rho, e = lindblad_solver(H, rho, time, *args, c_ops=[], e_ops=e_ops)
        if len(e_ops):
            e_total.append(e[:, -1])

    if len(e_ops):
        # Careful, last value not being appended.
        e_total = np.array(e_total)
        e_total = np.moveaxis(e_total, 0, -1)
        e_total.reshape((len(e_ops), -1))

        # return time_total, e_total
        return e_total
    else:
        return (np.trace(rho @ np.kron(sx, si)) + 1) / 2
