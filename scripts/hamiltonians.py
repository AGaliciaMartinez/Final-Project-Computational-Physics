import numpy as np
import sys
sys.path.append('../scripts/')
from utils import sx, sy, sz, si, init_qubit

def single_carbon_H(t, wL, wh, theta):
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

def pair_carbons_H(t, X, Z):
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
