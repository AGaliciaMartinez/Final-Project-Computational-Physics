import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit, pi_rotation
from dynamical_decoupling import dynamical_decoupling


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


if __name__ == "__main__":
    steps = 5
    args = [1.0, 0.1, np.pi / 4]
    N, tau = pi_rotation(args[0], args[1], args[2])
    e_ops = [np.kron(si, sx), np.kron(si, sy), np.kron(si, sz)]

    rho_0 = np.kron(init_qubit([0, 0, 1]), init_qubit([0, 0, 1]))

    results1 = dynamical_decoupling(H,
                                    rho_0,
                                    N,
                                    tau,
                                    steps,
                                    args[0],
                                    args[1],
                                    args[2],
                                    e_ops=e_ops)

    rho_1 = np.kron(init_qubit([0, 0, -1]), init_qubit([0, 0, 1]))
    results2 = dynamical_decoupling(H,
                                    rho_1,
                                    N,
                                    tau,
                                    steps,
                                    args[0],
                                    args[1],
                                    args[2],
                                    e_ops=e_ops)

    evens = np.arange(0, N, 2)

    px1 = np.take(results1[0], evens)
    py1 = np.take(results1[1], evens)
    pz1 = np.take(results1[2], evens)

    px2 = np.take(results2[0], evens)
    py2 = np.take(results2[1], evens)
    pz2 = np.take(results2[2], evens)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.set_title('Rotation about -x axis')
    ax1.set_ylabel('Projections')
    ax1.plot(evens, px1, label='X', color='blue')
    ax1.plot(evens, py1, label='Y', color='red')
    ax1.plot(evens, pz1, label='Z', color='green')
    ax1.legend()

    ax2.set_title('Rotation about x axis')
    ax2.set_ylabel('Projections')
    ax2.plot(evens, px2, label='X', color='blue')
    ax2.plot(evens, py2, label='Y', color='red')
    ax2.plot(evens, pz2, label='Z', color='green')
    ax2.legend()

    plt.ylabel('Projections')
    plt.xlabel('Number of Dyn. Decoupling Sequences')

    plt.show()
