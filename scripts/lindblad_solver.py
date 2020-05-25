import numpy as np
from matplotlib import pyplot as plt
from utils import si, sx, sy, sz, init_qubit

sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)


def _lindblad(H, rho, t, c_ops):
    """Return the evaluation of the Linbald operator."""
    lind = -1j * (H(t) @ rho - rho @ H(t))
    for op in c_ops:
        lind += op @ rho @ np.conj(op) - 1 / 2 * (np.conj(op) @ op @ rho +
                                                  rho @ np.conj(op) @ op)
    return lind


def _runge_kutta(H, rho_last, t, dt, c_ops):
    """Perfor an integration step using the runge kutta 4 algorithm."""

    k1 = _lindblad(H, rho_last, t, c_ops)
    k2 = _lindblad(H, rho_last + dt / 2 * k1, t + dt / 2, c_ops)
    k3 = _lindblad(H, rho_last + dt / 2 * k2, t + dt / 2, c_ops)
    k4 = _lindblad(H, rho_last + dt * k3, t + dt, c_ops)

    rho_next = rho_last + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)

    return rho_next


def lindblad_solver(H, rho_0, tlist, *args, c_ops=[], e_ops=[]):
    """Main solver for the Linbald equation. It uses Runge Kutta 4 to solve the
    ordinari differential equations.

    Input:

    H - the Hamiltonian of the system to be evolved

    rho_0 - initial state (must be a density matrix)

    tlist - the list of times over which to evolve the system

    c_ops - collapse operators

    e_ops - desired expectation value operators


    Output:

    rho_f - the final density matrix of the system

    expectations - the expectation values at each time step

    """

    # Allocation of arrays
    expectations = np.zeros((len(e_ops), len(tlist)))
    rho_last = rho_0

    # Evaluate expectation values
    for num, op in enumerate(e_ops):
        expectations[num, 0] = np.trace(rho_last @ op)

    for i in range(1, len(tlist)):
        dt = tlist[i] - tlist[i - 1]
        rho_next = _runge_kutta(H, rho_last, tlist[i], dt, c_ops)

        # Evaluate expectation values
        for num, op in enumerate(e_ops):
            expectations[num, i] = np.trace(rho_next @ op)

        rho_last = rho_next
    rho_f = rho_last

    return rho_f, expectations


if __name__ == "__main__":
    Ham = sz

    def H(t, frequency):
        return Ham * frequency

    rho_0 = init_qubit([1, 0, 0])
    tlist = np.linspace(0, 100, 1000)
    frequency = 1

    rho, expect = lindblad_solver(H,
                                  rho_0,
                                  tlist,
                                  frequency,
                                  c_ops=[np.sqrt(0.05) * sz],
                                  e_ops=[si, sx, sy, sz])

    plt.plot(tlist, expect[0, :], label='I')
    plt.plot(tlist, expect[1, :], label='X')
    plt.plot(tlist, expect[2, :], label='Y')
    plt.plot(tlist, expect[3, :], label='Z')
    plt.legend()
    plt.show()
