
from matplotlib import pyplot as plt
from implementation.tps import _gain_corrector, find_gated_area, find_tps_height, trap_filter, exp_func
import numpy as np
from exp import second_ord_exp_decay, first_ord_exp_decay
from tools.plot_tools import plot_io, plot_several, plot_traps


import warnings

warnings.filterwarnings("error")  # Converte i warning in eccezioni
# Esempio di utilizzo della funzione
dd = 0.01
N = 4000
H1_star=2.0
M = 10

sim_bias=1

# Array dei campioni
nn = np.arange(0, N)
tt = nn * dd

# Array della forma d'onda di input
t_start = 10 * dd

# v1 = first_ord_exp_decay(tt, np.zeros_like(tt), t_start, H1_star,0,M,0,0)


# Parametri del filtro trapezoidale
m = 200
l = 100

v_s = []
# v_s.append(v1)
tau1_v=[]
# tau1_v.append(0)
H1_v=[]
# H1_v.append(H1_star)
tau1 = 0.01
sigma_v=[]

# ########## no erf
# for i in range (1,3):
#     sigma=0
#     tau1 *= 10
#     tau1_v.append(tau1)
#     H1=H1_star
#     H1_v.append(H1)
#     sigma_v.append(sigma)
#     v_s.append(second_ord_exp_decay(tt, np.zeros_like(tt), t_start, H1, tau1, M, sigma, 0))
# ########## no erf

######### erf
sigmasss=[1e-5, 1e-3]
for i in range (1,3):
    tau1 *= 2
    sigma=sigmasss[i-1]
    tau1_v.append(tau1)
    H1=H1_star
    H1_v.append(H1)
    sigma_v.append(sigma)
    v_s.append(second_ord_exp_decay(tt, np.zeros_like(tt), t_start, H1, tau1, M, sigma, 0))
########## erf
# v2 = exp_func(tt, H2, t2, M)
# vv = v1 

s_vals_scaled=[]



for v, H1 in zip(v_s, H1_v):
    _, _, _ , sss = trap_filter(M, dd, v, m, l)
    print(H1)
    print(-np.max(v)+H1)
    s_vals_scaled.append(sss)

int_w=10
predelay=100
width=400

# tps_height=find_tps_height(base_mean=0, w=[round(t1)+l, round(t1)+l+m], s_vals_scaled=s_vals_scaled)
# tps_area=find_gated_area(base_mean=0, t=[t1-predelay,t1-predelay+width], input=vv, dt=dd)

max_s= np.max(s_vals_scaled)
print("max trap out: ", max_s)
# max_v= np.max(vv)
# print("max trap in: ", max_v)
print("H1: ",H1_v)
# print("estimated tps height: ", tps_height)
print("trap scaled in units of gain:", 1/_gain_corrector(M/dd, l))
plot_several(tt=tt, input=v_s, out_scaled=s_vals_scaled, peaks_signal=0, t_zeros=None, t_end=None, tau1=tau1_v, nominal_value=H1_star, sigma=sigma_v)
# for vi,t1 in zip(v_s, tau1):
#     plot_traps(tt=tt, input=vi, out_scaled=[s_vals_scaled,], peaks_signal=dml_vals, t_zeros=None, t_end=None, tau1=t1)
