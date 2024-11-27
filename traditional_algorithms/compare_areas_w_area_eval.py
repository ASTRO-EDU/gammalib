import sys
import csv

from implementation.tps import compute_and_separate_trap, plot_traps, trap_filter, exp_func, find_area
import numpy as np
import matplotlib.pyplot as plt

from gammasim import GammaSim
import area_eval

config_file="config_method2-w-noise.json"
save_output_fig="out_no_noise/"
saturation = False
gammasim = GammaSim(config_file)
gammasim.generate_dataset(saturation)

vv_d = gammasim.get_dataset()

# Esempio di utilizzo della funzione
dd = 1
N = vv_d.shape
print(N)

# H1= gammasim.get_params()[0][0]['gamma']
# t1 = gammasim.get_params()[0][0]['t_start']

tt = np.arange(0, vv_d.shape[1])

avg=0
n=0
sigmaq = 0  # For tracking the sum of squared differences (variance)

# Read parameters and avg area ratio from the CSV file
with open('trap_params.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        m = int(row[0])  # First column is m
        l = int(row[1])  # Second column is l
        avg_area_ratio_gain = float(row[2])  # Third column is avg area ratio
        variance = float(row[3])  # Fourth column is variance
        print(f"m: {m}, l: {l}, pre-computed avg area ratio gain: {avg_area_ratio_gain}, Variance: {variance}")

a_output = np.zeros((N[0], 1))

i=0
random_plot=np.random.randint(vv_d.shape[1])
for vv in vv_d:

    # Chiamata alla funzione
    M = -1/(np.log(.01)/gammasim.get_params()[i][0]['tau2'])
    dml_vals, p_vals, s_vals, s_scaled = trap_filter(M, dd, vv, m, l)
    if i==random_plot:
        plot_traps(tt=tt, input=vv, out_scaled=[s_vals/avg_area_ratio_gain,], peaks_signal=dml_vals, t_zeros=None, t_end=None)

    # v_opt = exp_func(tt, H1, t1, M)
    for j in range(1):
        a_output[i,j]=find_area(s_vals[j]/avg_area_ratio_gain, dd)
    i+=1

area_eval.plot_ARR(area_pred=a_output, area_real=gammasim.get_areas(),path=save_output_fig)
area_eval.plot_hists(area_pred=a_output, area_real=gammasim.get_areas(),path=save_output_fig)
area_eval.plot_gaussian_fitted(area_pred=a_output, area_real=gammasim.get_areas(),path=save_output_fig)

    
