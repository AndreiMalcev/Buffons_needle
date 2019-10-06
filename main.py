import numpy as np
import matplotlib.pyplot as plt
import math

D = 2
L = 1
MIN_N = 10
N = 10000
STEP_GRAPHIC = 10


def experiment_average(size):
    np.random.seed(size)
    h = np.random.uniform(0, D / 2, size=size)
    theta = np.random.uniform(0, math.pi, size=size)
    success = len(np.where((L * np.sin(theta) - 2 * h) >= 0)[0])
    return 2 * size * L / (success * D)


exp_vector = np.vectorize(experiment_average, otypes=[np.float], cache=False)
n_array = np.linspace(MIN_N, N, num=N - MIN_N + 1, dtype=np.int)
pi = np.array(exp_vector(n_array))
pi_mean = np.array([np.mean(pi[:i]) for i in range(1, N - MIN_N + 2)])
pi_delta = np.abs(pi_mean - math.pi)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.hlines(math.pi, xmin=0, xmax=N, colors='r', linewidth=2)
ax.plot(n_array[::STEP_GRAPHIC], pi_mean[::STEP_GRAPHIC], linewidth=1)
plt.ylim(3, 3.3)
plt.xlabel('N')
plt.ylabel('$\overline{\pi}_{average}$')
fig.savefig('C:\\Users\\user\\Desktop\\SPBU\\magistracy\\discrete and probabilistic models\\essay1\\fig\\pi.pdf')

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(n_array[::STEP_GRAPHIC], pi_delta[::STEP_GRAPHIC], linewidth=1, c='y')
plt.ylabel('$\Delta \overline{\pi}_{average}$')
plt.ylim(0, 0.1)
plt.xlabel('$N$')
ax.hlines(0, xmin=0, xmax=N)
ax.text(0.62, 0.95, '$min(\Delta \overline{\pi}_{average})=$' + '%.6f' % np.amin(pi_delta), transform=ax.transAxes)
fig.savefig('C:\\Users\\user\\Desktop\\SPBU\\magistracy\\discrete and probabilistic models\\essay1\\fig\\delta_pi.pdf')
