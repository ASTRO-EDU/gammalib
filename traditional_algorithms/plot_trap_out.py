
from implementation.tps import trap_filter, exp_func
import numpy as np
import matplotlib.pyplot as plt

# Esempio di utilizzo della funzione
dd = 1
N = 1000
H1=1
M = 600

# Array dei campioni
nn = np.arange(0, N)
tt = nn * dd

# Array della forma d'onda di input
t1 = 10 * dd
t2 = 4000 * dd
v1 = exp_func(tt, H1, t1, M)
v2 = exp_func(tt, 0, t2, M)
vv = v1 + v2

# Parametri del filtro trapezoidale
m = 200
l = 100

# Chiamata alla funzione
dml_vals, p_vals, s_vals , s_vals_scaled = trap_filter(M, dd, vv, m, l)

# Plot s_vals and vv against tt
plt.figure(figsize=(10, 6))

# Plot s_vals
plt.plot(tt, s_vals_scaled, label='s_vals', color='b', linewidth=2)

# Plot vv
plt.plot(tt, vv, label='vv', color='r', linewidth=2)

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('s_vals and vv vs. Time')

# Add legend
plt.legend()

# Display grid
plt.grid(True)

# Show the plot
plt.show()
