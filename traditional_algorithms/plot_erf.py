from matplotlib import pyplot as plt
import numpy as np
from exp import second_ord_exp_decay, first_ord_exp_decay

# Esempio di utilizzo della funzione
dd = 0.001
N = 4000
H1_star=1
M = 1

sim_bias=1

# Array dei campioni
nn = np.arange(0, N)
tt = nn * dd

# Array della forma d'onda di input
t_start = 10 * dd

tau1 = 0.001
sigma = np.linspace(0, 0.005, 10)
# for s in sigma:
#     fun=first_ord_exp_decay(tt, np.zeros_like(tt), t_start, H1_star, 0, M, s, 0)

#     # Plot vv
#     # plt.plot(tt, peaks_signal, label='dml_vals', color='r', linewidth=2)
#     plt.plot(tt, fun, label=f"first ord, sigma: {s}", linestyle="--", linewidth=2)
for s in sigma:
    fun=second_ord_exp_decay(tt, np.zeros_like(tt), t_start, H1_star, tau1, M, s, 0)

    # Plot vv
    # plt.plot(tt, peaks_signal, label='dml_vals', color='r', linewidth=2)
    plt.plot(tt, fun, label=f"second ord, sigma: {s}", linestyle="--", linewidth=2)

    
        
# plt.plot(tt, output, label='s_vals', color='b', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('s_vals and vv vs. Time')

plt.legend()

plt.grid(True)

plt.show()

