import numpy as np
from matplotlib import pyplot as plt

sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)


def lindblad(H, rho, t, c_ops):
    lind = -1j * (H(t) @ rho - rho @ H(t))
    for op in c_ops:
        lind += np.conj(op) @ rho @ op - 1 / 2 * (np.conj(op) @ op @ rho -
                                                  rho @ np.conj(op) @ op)
    return lind


def runge_kutta(H, rho_last, t, dt, c_ops):

    k1 = lindblad(H, rho_last, t, c_ops)
    k2 = lindblad(H, rho_last + dt / 2 * k1, t + dt / 2, c_ops)
    k3 = lindblad(H, rho_last + dt / 2 * k2, t + dt / 2, c_ops)
    k4 = lindblad(H, rho_last + dt * k3, t + dt, c_ops)

    rho_next = rho_last + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)

    return rho_next


def lindblad_solver(H, rho_0, tlist, c_ops=[], e_ops=[], *args):
    """
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

    # defining the necessary lists to store expectation values and states
    expectations = np.zeros((len(tlist), len(e_ops)))
    rho_last = rho_0
    # print(rho_last)
    for num, op in enumerate(e_ops):
        expectations[0, num] = np.trace(rho_last @ op)

    print(expectations[0, :])

    for i in range(1, len(tlist)):
        # # euler integration method
        # f = -1j * (H @ rho_last - rho_last @ H)
        # rho_next = f * (tlist[i] - tlist[i - 1]) + rho_last

        dt = tlist[i] - tlist[i - 1]
        rho_next = runge_kutta(H, rho_last, tlist[i], dt, c_ops)

        for num, op in enumerate(e_ops):
            expectations[i, num] = np.trace(rho_next @ op)

        rho_last = rho_next
    rho_f = rho_last

    return rho_f, expectations


if __name__ == "__main__":
    Ham = np.array([[1, 0], [0, -1]])

    def H(t):
        return Ham

    rho_0 = 1 / 2 * np.array([[1, 1], [1, 1]], dtype=complex)
    tlist = np.linspace(0, 100, 1000)

    rho, expect = lindblad_solver(H,
                                  rho_0,
                                  tlist,
                                  c_ops=[np.sqrt(0.05) * sigmaz],
                                  e_ops=[np.eye(2), sigmax, sigmay, sigmaz])

    # plt.plot(tlist, expect[:, 0], label='I')
    plt.plot(tlist, expect[:, 1], label='X')
    plt.plot(tlist, expect[:, 2], label='Y')
    plt.plot(tlist, expect[:, 3], label='Z')
    plt.legend()
    plt.show()
