import numpy as np
import matplotlib.pyplot as plt
from configuration.model.config_model import load_config
from gammasim import GammaSim

from implementation.tps import compute_height_window, detect_waveforms, find_base_mean, find_tps_height, time_filter, trap_filter
from tools.plot_tools import plot_input_trap_time_waveforms

cmap = plt.get_cmap('viridis')  # Use a colormap (e.g., 'viridis')

# Specifica il nome del file di configurazione

config_file_name = "config_tps_config_method2-w-noise.json"  
# config_file_name = "config_tps_config_method2-w-noise.json"  

####

# Carica la configurazione
cfg = load_config(config_file_name)
print(f"Configuration imported! \n {cfg} \n\n")
##Istanzio il simulatore
config_file=cfg.gammasim_cfg
saturation = False
gammasim = GammaSim(config_file)

gammasim.generate_dataset(saturation)

###prendo i dati - input 
vv = gammasim.get_dataset()[0]
print(vv.shape)

dd = gammasim.get_sampling_time()
M = gammasim.get_params()[0][0]['tau2']
print(f"Sampling: {dd}s, decay-time constant: {M}")
H1= gammasim.get_params()[0][0]['gamma']
t1 = gammasim.get_params()[0][0]['t_start']
### __init__
N = vv.shape[0]
nn = np.arange(0, N)
tt = nn * dd

################ Method: Compute Trapezoidal shaper. Return heights
trap_heights = []
trap_area = []
first_iteration=True
base_mean=0
scaled_tps_out=None

s_vals=0
y_l , y_h, dy, d2y = time_filter(
    tt, vv, 
    cfg.time_filter.alpha_l, 
    cfg.time_filter.alpha_h, 
    cfg.time_filter.gain_k, 
    cfg.time_filter.in_cond
    )

d2y = np.convolve(d2y, np.ones(cfg.time_filter.window_sz)/cfg.time_filter.window_sz, mode='same')

t_zeros = detect_waveforms(
    dyy=dy, d2yy=d2y, 
    der_ord_detection=cfg.time_filter.dev_ord_det, 
    th_detection_dy=cfg.time_filter.th_dy, 
    th_detection_d2y=cfg.time_filter.th_d2y,
    zero_crossing_window=cfg.time_filter.zero_cr_window)

top_mean_w = compute_height_window(
    t_zeros, 
    m=cfg.trap_filter.m, 
    l=cfg.trap_filter.l, 
    ftd_s=cfg.trap_filter.ftd_s, 
    ftd_e=cfg.trap_filter.ftd_e, 
    )

print(f"t zeros: {t_zeros}")

if len(t_zeros)>0:
    dml_vals, p_vals, s_vals , scaled_tps_out = trap_filter(M, dd, vv, cfg.trap_filter.m, cfg.trap_filter.l)
    
    try:
        base_mean=find_base_mean(scaled_tps_out, t_zeros[0], cfg.trap_filter.m)
        
    except ValueError as e:
        if first_iteration:
            raise ValueError(f"Error during first iteration: {str(e)}") from e
    j=0
    for w in top_mean_w:
        height = find_tps_height(base_mean=base_mean, w=w, s_vals_scaled=scaled_tps_out)
        trap_heights.append(height)

    
else:
    print("warning: no detection")

print("base mean:  ", base_mean)
print("height: ", trap_heights)
print("height: ", gammasim.get_heights()[0])

################ #

plot_input_trap_time_waveforms(
    nn, 1, vv, 
    scaled_tps_out, top_mean_w, trap_heights, base_mean, 
    cfg.time_filter.th_dy, None, y_l, y_h, dy, d2y, 
    cfg.time_filter.alpha_l,  cfg.time_filter.alpha_h
    )