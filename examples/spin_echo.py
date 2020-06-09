import numpy as np
import parmap
from matplotlib import pyplot as plt
from itertools import product

import sys
sys.path.append('../scripts/')
from lindblad_solver import lindblad_solver
from utils import sx, sy, sz, si, init_qubit, normal_autocorr_generator
from dynamical_decoupling import dynamical_decoupling


def H(t, dw_it=[]):
    return -next(dw_it) * sz


def dd_wrapper(H, tau_list, dt, N, mu, sigma, corr_time, seed):
    N = 1
    e = []

    # Initial state
    rho_0 = init_qubit([1, 0, 0])

    for tau_final in tau_list:
        dw_it = normal_autocorr_generator(mu, sigma, corr_time / dt / 2, seed)
        tau = np.arange(tau_final, step=dt)
        e.append(
            dynamical_decoupling(H, rho_0, N, tau_final, tau.shape[0], dw_it))
    return e


dt = [0.2]
N = [1]
sigma = [0.5, 1, 2]
mu = [0]
corr_time = [1000]

repetitions = 1000
n_tau = 10
tau_list = [np.linspace(1, 20, n_tau)]
# seed_list = np.arange(n_tau * repetitions).reshape((repetitions, n_tau))
seed_list = np.arange(repetitions)

values = list(product([H], tau_list, dt, N, mu, sigma, corr_time, seed_list))

results = parmap.starmap(dd_wrapper, values, pm_chunksize=3, pm_pbar=True)

results = np.array(results)
print(results.shape)

# Adapt results to input
results = results.reshape((3, repetitions, n_tau))

results_mean = results.mean(axis=-2)
results_std = results.std(axis=-2) / np.sqrt(repetitions - 1)

plt.errorbar(tau_list[0], results_mean[0, :], results_std[0, :])
plt.errorbar(tau_list[0], results_mean[1, :], results_std[1, :])
plt.errorbar(tau_list[0], results_mean[2, :], results_std[1, :])
plt
# plt.errorbar(tau_list[0], results_mean, results_std)
plt.show()
