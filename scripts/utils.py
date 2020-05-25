import numpy as np
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
si = np.eye(2, dtype=complex)


def init_qubit(bloch_direction):
    """Returns a qubit with the bloch vector pointing in the given direction. n
    must be a numpy array and it is not required to be normalized (it is
    normalized before creating the state)

    """
    n = np.linalg.norm(bloch_direction)
    return (si + sx * n[0] + sy * n[1] + sz * n[2]) / 2
