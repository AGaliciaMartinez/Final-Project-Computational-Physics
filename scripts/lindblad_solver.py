import numpy as np
from matplotlib import pyplot as plt
from utils import si, sx, sy, sz, init_qubit


def _lindblad(H, rho, c_ops):
    """Return the evaluation of the Linbald operator."""
    lind = -1j * (H @ rho - rho @ H)
    # TODO Change this to numpy cast rules as we did for the evaluation of the
    # operators. (implement first a test function)
    for op in c_ops:
        lind += op @ rho @ op.conj() - 1 / 2 * (op.conj() @ op @ rho +
                                                rho @ op.conj() @ op)
    return lind


def _runge_kutta_generator(H, rho, tlist, c_ops, *args):
    """
    Perform an integration step using the Runge-Kutta 4th order algorithm.

    Input:

    H - the Hamiltonian of interest for the decoupling series

    rho - the initial state of the system

    tlist - the list of times to integrate for, this function only
    computes one step at a time but tlist is an argument for
    consistency with lindblad_solver function

    c_ops - collapse operators

    *args - extra arguments passed to the Hamiltonian.

    Output:
    State of the system at the next time step.
    """
    H_t = H(tlist[1], *args)
    yield rho  # Iteration starts with the initial state for simplicity

    for i, t in enumerate(tlist[1:], 1):
        dt = tlist[i] - tlist[i - 1]

        H_t_dt_2 = H(t + dt / 2, *args)
        H_t_dt = H(t + dt, *args)

        k1 = _lindblad(H_t, rho, c_ops)
        k2 = _lindblad(H_t_dt_2, rho + dt / 2 * k1, c_ops)
        k3 = _lindblad(H_t_dt_2, rho + dt / 2 * k2, c_ops)
        k4 = _lindblad(H_t_dt, rho + dt * k3, c_ops)

        H_t = H_t_dt

        rho = rho + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        yield rho


def lindblad_solver(H, rho, tlist, *args, c_ops=[], e_ops=[]):
    """Main solver for the Linbald equation. It uses Runge Kutta 4 to solve the
    ordinary differential equations.

    Input:

    H - the Hamiltonian of the system to be evolved

    rho - initial state (must be a density matrix)

    tlist - the list of times over which to evolve the system

    *args - extra arguments passed to the Hamiltonian.

    c_ops - collapse operators

    e_ops - desired expectation value operators


    Output:

    rho_f - the final density matrix of the system

    expectations - the expectation values at each time step. Size =
    (len(e_ops), len(tlist))

    """
    # Allocation of arrays
    expectations = np.zeros((len(e_ops), len(tlist)))
    e_ops_np = np.array(e_ops)

    rk_iterator = _runge_kutta_generator(H, rho, tlist, c_ops, *args)

    for i, rho in enumerate(rk_iterator):
        if len(e_ops):  # Compute all the expectations if any
            expectations[:, i] = np.trace(rho @ e_ops_np, axis1=1, axis2=2)

    return rho, expectations


if __name__ == "__main__":

    def H(t, frequency):
        return sz * frequency / 2

    rho_0 = init_qubit([1, 0, 0])
    tlist = np.linspace(0, 3.2, 1000)

    frequency = 1 * 2 * np.pi
    rho, expect = lindblad_solver(
        H,
        rho_0,
        tlist,
        frequency,  # Extra argument to H
        c_ops=[np.sqrt(0.5) * sz],
        e_ops=[si, sx, sy, sz])

    rho = sx @ rho @ sx

    rho, expect_2 = lindblad_solver(
        H,
        rho,
        tlist,
        frequency,  # Extra argument to H
        c_ops=[np.sqrt(0.5) * sz],
        e_ops=[si, sx, sy, sz])

    print(expect.shape, expect_2.shape)

    expect = np.concatenate((expect, expect_2), axis=1)
    tlist = np.linspace(0, 6.4, 2000)

    plt.plot(tlist, expect[0, :], label='I')
    plt.plot(tlist, expect[1, :], label='X')
    plt.plot(tlist, expect[2, :], label='Y')
    plt.plot(tlist, expect[3, :], label='Z')
    plt.plot(tlist, np.exp(-tlist))
    plt.legend()
    plt.show()
