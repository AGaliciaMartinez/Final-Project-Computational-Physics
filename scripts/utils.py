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

    if type(bloch_direction) == type([]):
        n = np.array(bloch_direction)

    else:
        n = bloch_direction

    if np.linalg.norm(n) != 0:
        n = n / np.linalg.norm(n)
    return (si + sx * n[0] + sy * n[1] + sz * n[2]) / 2


def pi_rotation(wL, wh, theta):
    tau = np.pi / (2 * wL + wh * np.cos(theta))

    A = wh * np.cos(theta)
    B = wh * np.sin(theta)
    w_tilde = np.sqrt((A + wL)**2 + B**2)
    mz = (A + wL) / w_tilde
    mx = B / w_tilde
    alpha = w_tilde * tau
    beta = wL * tau
    phi = np.arccos(mz * np.sin(alpha) * np.sin(beta) -
                    np.cos(alpha) * np.cos(beta))

    N = int(round(np.pi / np.abs(phi)))
    return N, tau, phi


def normal_autocorr_generator(mu, sigma, tau, seed):
    """Returns an iterator that is used to generate an autocorrelated sequence of
    Gaussian random numbers.

    Each of the random numbers in the sequence is distributed
    according to a Gaussian with mean `mu` and standard deviation `sigma` (just
    as in `numpy.random.normal`, with `loc=mu` and `scale=sigma`). Subsequent
    random numbers are correlated such that the autocorrelation function
    is on average `exp(-n/tau)` where `n` is the distance between random
    numbers in the sequence.

    This function implements the algorithm described in
    https://www.cmu.edu/biolphys/deserno/pdf/corr_gaussian_random.pdf

    Parameters
    ----------

    mu: float
        mean of each Gaussian random number
    sigma: float
        standard deviation of each Gaussian random number
    tau: float
        autocorrelation time

    Returns:
    --------
    sequence: numpy array
        array of autocorrelated random numbers

    """
    f = np.exp(-1. / tau)

    rng = np.random.default_rng(seed=seed)
    sequence = rng.normal(0, 1)

    while True:
        sequence = f * sequence + np.sqrt(1 - f**2) * rng.normal(0, 1)
        yield mu + sigma * sequence
