import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts/')
from plot_utils import set_size
from utils import sx, sy, sz, si, init_qubit, pi_rotation
from dynamical_decoupling import dynamical_decoupling
from hamiltonians import single_carbon_H

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 14,
    "font.size": 16,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}

mpl.rcParams.update(nice_fonts)

if __name__ == "__main__":
    steps = 5
    args = [1.0, 0.1, np.pi / 4]
    N, tau, phi = pi_rotation(args[0], args[1], args[2])
    e_ops = [np.kron(si, sx), np.kron(si, sy), np.kron(si, sz)]

    rho_0 = np.kron(init_qubit([0, 0, 1]), init_qubit([0, 0, 1]))

    _, _, results1 = dynamical_decoupling(single_carbon_H,
                                          rho_0,
                                          N,
                                          tau,
                                          steps,
                                          args[0],
                                          args[1],
                                          args[2],
                                          e_ops=e_ops)

    rho_1 = np.kron(init_qubit([0, 0, -1]), init_qubit([0, 0, 1]))
    _, _, results2 = dynamical_decoupling(single_carbon_H,
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

    print(pz1[-1])
    print(pz2[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(width='report_full'))

    plt.figure(1)
    ax1.set_title(r'Rotation of Carbon with NV in $|0\rangle$ state')
    ax1.set_ylabel('Projections')
    ax1.plot(evens, px1, label='X', color='blue')
    ax1.plot(evens, py1, label='Y', color='red')
    ax1.plot(evens, pz1, label='Z', color='green')
    ax1.legend()

    ax2.set_title(r'Rotation of Carbon with NV in $|1\rangle$ state')
    ax2.set_ylabel('Projections')
    ax2.plot(evens, px2, label='X', color='blue')
    ax2.plot(evens, py2, label='Y', color='red')
    ax2.plot(evens, pz2, label='Z', color='green')
    ax2.legend()

    plt.ylabel('Projections')
    plt.xlabel('Number of Dyn. Decoupling Sequences')

    plt.show()
