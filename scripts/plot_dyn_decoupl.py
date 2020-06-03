import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
from scipy.optimize import curve_fit
import sys
sys.path.append('../scripts/')
from plot_utils import set_size

data = np.load('../script_output/data_dyn_decoupl0.npz')

taus = data["taus"]
proj1 = data["proj1"]
proj2 = data["proj2"]


plt.plot(taus, proj1, label='q1')
plt.plot(taus, proj2, label='q2')
plt.ylabel(r'$P_x$')
plt.xlabel(r'$\tau$')
plt.tight_layout()
plt.legend()

plt.show()
