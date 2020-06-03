import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit
from lindblad_solver import lindblad_solver
from tqdm import tqdm
from itertools import repeat
import multiprocessing as mp


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


def dynamical_decoupling(H, rho_0, N, tau, steps, *args):
    """
    Input:
    tau - the free evolution time in the dynamical decoupling sequence
    described by tau - R(pi) - 2tau - R(pi) - tau pulses

    Output:
    Returns the projection along the x axis of the eletron's state after
    N decoupling sequences.
    """
    time1 = np.linspace(0, tau, steps)
    time2 = np.linspace(tau, 3 * tau, 2 * steps)
    time3 = np.linspace(3 * tau, 4 * tau, steps)

    # initial density matrix for ms=0
    rho_last = rho_0

    # implement N dynamical decoupling cycles
    for i in range(N):
        # tau evolution
        rho, _ = lindblad_solver(H, rho_last, time1, *args, c_ops=[], e_ops=[])
        rho_down = np.kron(sx, si) @ rho @ np.kron(sx, si)
        # 2 tau flipped evolution
        rho_fl, _ = lindblad_solver(H,
                                    rho_down,
                                    time2,
                                    *args,
                                    c_ops=[],
                                    e_ops=[])
        rho_up = np.kron(sx, si) @ rho_fl @ np.kron(sx, si)
        # tau flipped back evolution
        rho_flb, _ = lindblad_solver(H,
                                     rho_up,
                                     time3,
                                     *args,
                                     c_ops=[],
                                     e_ops=[])

        rho_last = rho_flb

        time1 = time1 + 4 * tau
        time2 = time2 + 4 * tau
        time3 = time3 + 4 * tau

    exp = [
        np.trace(rho_last @ np.kron(si, sx)),
        np.trace(rho_last @ np.kron(si, sy)),
        np.trace(rho_last @ np.kron(si, sz))
    ]
    return exp[0], exp[1], exp[2]
    # return np.trace(rho_last @ np.kron(sx, si))


if __name__ == "__main__":
    steps = 25
    args = [1.0, 0.1, np.pi / 4]

    tau = 1.517
    N = np.arange(0, 33)

    rho_0 = np.kron((si + sz) / 2, (si + sz) / 2)
    rho_1 = np.kron((si - sz) / 2, (si + sz) / 2)

    parameters0 = zip(repeat(H), repeat(rho_0), N, repeat(tau), repeat(steps),
                      repeat(args[0]), repeat(args[1]), repeat(args[2]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results0 = pool.starmap(dynamical_decoupling, parameters0)

    px = np.zeros(len(N))
    py = np.zeros(len(N))
    pz = np.zeros(len(N))
    for i in N - 1:
        px[i] = results0[i][0]
        py[i] = results0[i][1]
        pz[i] = results0[i][2]

    parameters1 = zip(repeat(H), repeat(rho_1), N, repeat(tau), repeat(steps),
                      repeat(args[0]), repeat(args[1]), repeat(args[2]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results1 = pool.starmap(dynamical_decoupling, parameters1)

    px1 = np.zeros(len(N))
    py1 = np.zeros(len(N))
    pz1 = np.zeros(len(N))
    for i in N - 1:
        px1[i] = results1[i][0]
        py1[i] = results1[i][1]
        pz1[i] = results1[i][2]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.set_title('Rotation about -x axis')
    ax1.set_ylabel('Projections')
    ax1.plot(N, px, label='X', color='blue')
    ax1.plot(N, py, label='Y', color='red')
    ax1.plot(N, pz, label='Z', color='black')
    ax1.legend()

    ax2.set_title('Rotation about x axis')
    ax2.set_ylabel('Projections')
    ax2.plot(N, px1, label='X', color='blue')
    ax2.plot(N, py1, label='Y', color='red')
    ax2.plot(N, pz1, label='Z', color='black')
    ax2.legend()

    plt.ylabel('Projections')
    plt.xlabel('Number of Dyn. Decoupling Sequences')

    plt.show()
