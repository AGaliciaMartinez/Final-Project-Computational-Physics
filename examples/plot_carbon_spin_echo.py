import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('../scripts/')
from plot_utils import set_size

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 14,
    "font.size": 14,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

mpl.rcParams.update(nice_fonts)

data1 = np.load('../script_output/protecting_carbon.npz')
data2 = np.load('../script_output/protecting_carbon_sequences_2.npz')
data4 = np.load('../script_output/protecting_carbon_sequences_4.npz')
data8 = np.load('../script_output/protecting_carbon_sequences_8.npz')
data16 = np.load('../script_output/protecting_carbon_sequences_16.npz')
data16_glued = np.load(
    '../script_output/protecting_carbon_sequences_16_glued.npz')

time1 = data1['time']
exp_mean1 = (data1['exp_mean'] + 1) / 2
exp_std1 = data1['exp_std'] / 2

time2 = data2['time']
exp_mean2 = data2['exp_mean']
exp_std2 = data2['exp_std']

time4 = data4['time']
exp_mean4 = data4['exp_mean']
exp_std4 = data4['exp_std']

time8 = data8['time']
exp_mean8 = data8['exp_mean']
exp_std8 = data8['exp_std']

time16 = np.concatenate((data16['time'], data16_glued['time']))
exp_mean16 = np.concatenate((data16['exp_mean'], data16_glued['exp_mean']))
exp_std16 = np.concatenate((data16['exp_std'], data16_glued['exp_std']))


def fitter(x, T, n, a):
    return (0.5 * np.exp(-np.power(x / T, n))) * a + 0.5


popt1, pcov1 = curve_fit(fitter, time1, exp_mean1, p0=(750, 3, 1))
popt2, pcov2 = curve_fit(fitter, time2, exp_mean2, p0=(750, 3, 1))
popt4, pcov4 = curve_fit(fitter, time4, exp_mean4, p0=(750, 3, 1))
popt8, pcov8 = curve_fit(fitter, time8, exp_mean8, p0=(750, 3, 1))
popt16, pcov16 = curve_fit(fitter, time16, exp_mean16, p0=(750, 3, 1))
print(popt2[0])
print(pcov2[0, 0])

times = np.logspace(0, 4, 100000)
fig, ax = plt.subplots(1, 1, figsize=set_size(width='report_full'))

# plt.errorbar(time1, exp_mean1, exp_std1, label='N=1', color='black')
# plt.plot(times, fitter(times, *popt1), '--', color='black')
plt.errorbar(time2, exp_mean2, exp_std2, label='N=2', color='green')
plt.plot(times, fitter(times, *popt2), 'g--')
plt.errorbar(time4, exp_mean4, exp_std4, label='N=4', color='blue')
plt.plot(times, fitter(times, *popt4), 'b--')
plt.errorbar(time8, exp_mean8, exp_std8, label='N=8', color='orange')
plt.plot(times, fitter(times, *popt8), '--', color='orange')
plt.errorbar(time16, exp_mean16, exp_std16, label='N=16', color='red')
plt.plot(times, fitter(times, *popt16), 'r--')
plt.title('Dynamical Decoupling of Carbon atoms')
plt.ylabel('Projection on x-axis')
plt.xlabel(r'$\tau$')
plt.xscale('log')
plt.xlim(40, 6000)
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
N = [2, 4, 8, 16]
Ts = [popt2[0], popt4[0], popt8[0], popt16[0]]
Ts_err = [pcov2[0, 0], pcov4[0, 0], pcov8[0, 0], pcov16[0, 0]]
plt.errorbar(N, Ts, Ts_err)
plt.title('Improved Protection through Decoupling')
plt.ylabel(r'Decoherence Time ($T_2$)')
plt.xlabel('Number of Decoupling Sequences Applied')
plt.show()
