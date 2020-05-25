import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)


def lindblad_solver(H, rho_0, tlist, c_ops=[], e_ops=[], *args):
    """
    Input:

    H - the Hamiltonian of the system to be evolved

    rho_0 - initial state (must be a density matrix)

    tlist - the list of times over which to evolve the system

    c_ops - collapse operators

    e_ops - desired expectation value operators

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

        # runge-kutta method
        dt = tlist[i] - tlist[i - 1]
        k1 = -1j * (H @ rho_last - rho_last @ H)
        rho_aux = rho_last + dt / 2 * k1
        k2 = -1j * (H @ rho_aux - rho_aux @ H)
        rho_aux = rho_last + dt / 2 * k2
        k3 = -1j * (H @ rho_aux - rho_aux @ H)
        rho_aux = rho_last + dt * k2
        k4 = -1j * (H @ rho_aux - rho_aux @ H)

        rho_next = rho_last + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        for num, op in enumerate(e_ops):
            print(np.trace(rho_next @ op))
            expectations[i, num] = np.trace(rho_next @ op)

        rho_last = rho_next

    rho_f = rho_last

    return rho_f, expectations


if __name__ == "__main__":
    H = -0.1 * 2 * np.pi * np.array([[1, 0], [0, -1]])
    rho_0 = 1 / 2 * np.array([[1, 1], [1, 1]], dtype=complex)
    # rho_0 = np.array([[1, 0], [0, 0]])
    tlist = np.linspace(0, 100, 1000)

    rho, expect = lindblad_solver(H,
                                  rho_0,
                                  tlist,
                                  e_ops=[np.eye(2), sigmax, sigmay, sigmaz])

    # fig, axes = subplots(1, 1)
    plt.plot(tlist, expect[:, 0], label='I')
    plt.plot(tlist, expect[:, 1], label='X')
    plt.plot(tlist, expect[:, 2], label='Y')
    plt.plot(tlist, expect[:, 3], label='Z')
    plt.legend()
    plt.show()
