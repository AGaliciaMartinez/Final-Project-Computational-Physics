import numpy as np
from utils import sz, sx, sy, si, init_qubit
from numpy.testing import assert_almost_equal

from lindblad_solver import lindblad_solver


def test_only_hamiltonian_no_t():

    # Simple sigma z Hamiltonian
    H = lambda t: sz
    # Start in |+> state
    rho_0 = init_qubit([1, 0, 0])
    tlist = np.linspace(0, 10, 1000)

    rho, expect = lindblad_solver(H, rho_0, tlist, e_ops=[si, sx, sy, sz])

    expected_sx = np.cos(2 * tlist)
    expected_sy = np.sin(2 * tlist)
    expected_sz = np.zeros_like(tlist)

    assert_almost_equal(expect[0, :], 1, 3)
    assert_almost_equal(expect[1, :], expected_sx, 2)
    assert_almost_equal(expect[2, :], expected_sy, 2)
    assert_almost_equal(expect[3, :], expected_sz, 2)
