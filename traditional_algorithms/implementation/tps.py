import numpy as np 
from scipy.signal import lfilter, find_peaks

def exp_func(tt, H, t_shift, M):
    """Funzione esponenziale per il calcolo della forma d'onda"""
    
    return H * np.exp(-(tt - t_shift) / M) * (tt > t_shift)

def _gain_corrector(M, l):
    return 1/(M*l)

def trap_filter(M, dt, v, m, l):
    """
    Computes Discrete Trapezoidal shaper based on the method described in:

    V. T. Jordanov, "Digital techniques for real-time pulse shaping in radiation measurements",
    Nuclear Instruments and Methods in Physics Research 
    Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
        Volume 353, Issues 1-3, 1994.

    Args:
        M: Time constant 
        v: Time series vector.
        dt: Sampling time.
        m: duration of the trapezoidal flat top in samples 
        l: duration of the trapezoidal ramp in samples

    Returns:
        dml_vals ramps computation
        p_vals
        s_vals filter output
        s_vals*gain_corrector filter output scaled to match input height

    """
    # FIR filter (parte 1)
    m1 = max(m + 2 * l, m + l)
    m2 = max(m + l, l)
    N_order = max(m1, m2)

    N1 = np.zeros(N_order + 1)
    N1[0] = 1
    N1[m + l] = -1
    N1[l] = -1
    N1[m + 2 * l] = 1
    D1 = [1]

    dml_vals = lfilter(N1, D1, v)

    # Accumulatore su p (parte 2)
    N2 = [1]
    D2 = [1, -1]

    p_vals = lfilter(N2, D2, dml_vals)

    # Accumulatore su s (parte 3)
    N3 = [1]
    D3 = [1, -1]

    s_vals = lfilter(N3, D3, p_vals + dml_vals * M/dt)

    gain_corrector = _gain_corrector(M/dt,l)

    return dml_vals, p_vals, s_vals, s_vals*gain_corrector

def time_filter(t, signal, alpha_l, alpha_h, gain, initial_conditions=[0]):
    y_low = gain*lp_filter(signal, alpha_l, initial_conditions)
    y_high = hp_filter(y_low, alpha_h)
    dy_dt = np.gradient(y_high, t)
    d2y_dt = np.gradient(dy_dt, t)

    return y_low, y_high, dy_dt, d2y_dt
    
# def compute_and_separate_trap(M, dd, vv, m, l, h=0.1):
#     dml_vals, p_vals, s_vals, s_vals_corr = trap_filter(M, dd, vv, m, l)
    
#     # Finding peaks
#     t_zeros = find_peaks(dml_vals, height=[h], width=max(l - 2, 1), prominence=[h / 2], distance=m)[0]
#     t_ends = find_peaks(dml_vals, width=max(m - 2, 1), distance=m)[0]

#     # print(f"t0 {t_zeros}")
#     # print(f"tend {t_ends}")
    
#     s_vals_l = []
#     s_vals_corr_l = []

#     # Iterate over t_zeros, checking for overflow by limiting the range
#     for i in range(len(t_zeros)):
#         # Limit j to t_ends that are greater than t_zeros[i]
#         valid_ends = [t_end for t_end in t_ends if t_end > t_zeros[i]]
        
#         if not valid_ends:
#             print(f"Warning: pile-up detected at index {i}")
#             continue
        
#         for t_end in valid_ends:
#             if i < len(t_zeros) - 1 and t_end <= t_zeros[i + 1]:
#                 # Normal case
#                 s_vals_l.append(extract_trap(s_vals, t_zeros[i], t_end))
#                 s_vals_corr_l.append(extract_trap(s_vals_corr, t_zeros[i], t_end))
#             elif i == len(t_zeros) - 1:
#                 # Last segment, no t_zeros[i + 1] to compare to
#                 s_vals_l.append(extract_trap(s_vals, t_zeros[i], t_end))
#                 s_vals_corr_l.append(extract_trap(s_vals_corr, t_zeros[i], t_end))

#     return dml_vals, p_vals, s_vals_l, s_vals_corr_l, t_zeros, t_ends

def check_peaks(peaks_1, peaks_2, range_peak):
    # Extract peak indexes
    indexes_1 = peaks_1[0]
    indexes_2 = peaks_2[0]

    validated_peaks = []

    # Check if the second peaks output has values
    if len(indexes_2) == 0:
        print("No peaks detected in the second set.")
        return validated_peaks  # Return an empty list if no peaks in the second set

    # Check if a peak in the first set happens within the range after peaks in the second set
    for idx_2 in indexes_2:
        # Check for peaks in the first set within the range [idx_2+1, idx_2+range_after_peak]
        possible_peaks = indexes_1[(indexes_1 <= idx_2) & (indexes_1 > idx_2 - range_peak)]
        
        if len(possible_peaks) > 0:
            # Append only the idx_2 value if it has a match in indexes_1
            validated_peaks.append(idx_2)

    return validated_peaks

import numpy as np

def find_base_mean(tps_scaled_out, first_t_zero, stop_before_t0):
    if first_t_zero<1:
        raise ValueError(f"Cannot compute mean: first_t_zero is too low: {stop_before_t0}")
    elif stop_before_t0 > first_t_zero:
        raise ValueError(f"Cannot compute mean: first_t_zero {first_t_zero} is greater than t zero: {stop_before_t0}")
    else:
        return np.mean(tps_scaled_out[0:first_t_zero - stop_before_t0])

def update_ma_signal(old_mean, new_value, alpha):
    # Update base mean using alpha
    return (1 - alpha) * old_mean + alpha * new_value
        

def find_tps_height(base_mean, w, s_vals_scaled):
        
    return np.mean(s_vals_scaled[w[0]:w[1]]) - base_mean

def find_gated_area(base_mean, t, input, dt):
        
    return np.sum(input[t[0]:t[1]])*dt - base_mean

    
def extract_trap(s, start, end):
    x=np.zeros_like(s)
    x[start:end]=s[start:end]
    return x

def lp_filter(signal, k, initial_conditions):
    N=[0,k]
    D=[1, -(1-k)]
    initial_conditions
    filtered_signal, _ = lfilter(N, D, signal, zi=initial_conditions)
    return filtered_signal

def hp_filter(signal, alpha):
    gain=(1+alpha)/2
    N=[1,-1]
    D=[1, -alpha]
    return gain*lfilter(N, D, signal)

def update_recursive_avg(avg, new_sample, n):
    return avg + (new_sample - avg) / n

def update_recursive_std(sigmaq, avg_square, avg_update, new_sample, n):
    return sigmaq + avg_square - avg_update**2 + (new_sample**2 - sigmaq - avg_square)/ n

def compute_threshold(dyy, th_dy):
    """
    Computes the detection threshold based on the mean and standard deviation
    of the `dyy` array and a scaling factor `th_dy`.

    Parameters:
    dyy (np.array): The input signal array.
    th_dy (float): A scaling factor to adjust the threshold.

    Returns:
    float: The computed detection threshold.
    """

    # Calculate mean and standard deviation of the input signal `dyy`
    mean_value_dy = np.mean(dyy)
    std_dev_dy = np.std(dyy)

    # Compute the detection threshold
    th_detection_dy = th_dy * std_dev_dy + mean_value_dy

    # Print the results
    print("Mean value of dyy:", mean_value_dy)
    print("Standard deviation of dyy:", std_dev_dy)

    return th_detection_dy

def detect_waveforms(dyy, d2yy, der_ord_detection, th_detection_dy=None, th_detection_d2y=None, zero_crossing_window=5):
    # print("ord detection: ", der_ord_detection)
    if not der_ord_detection in [1,2]:
        raise ValueError(f"Detection method inserted: {der_ord_detection} is not 1 or 2")

    if der_ord_detection == 1:
        # candidate_peaks=find_peaks(x=dyy, height=th_detection_dy, prominence=[1000], distance= 50  )
        candidate_peaks=find_peaks(x=dyy, height=th_detection_dy, distance= 50  )

    elif der_ord_detection == 2:
        candidate_peaks=find_peaks(x=d2yy, height=th_detection_d2y )

    # print("candidate_peaks: ", candidate_peaks)

    candidate_peaks = candidate_peaks[0]

    t_zeros = []
    half_window=round(zero_crossing_window/2)

    if len(candidate_peaks)>0:
        for peak in candidate_peaks:
            for i in range(peak-half_window, peak+half_window):
                # print(half_window)
                if np.sign(d2yy[i]) != np.sign(d2yy[i + 1]):
                    
                    # Linear interpolation to estimate zero-crossing point
                    t_zeros.append(peak)
                    # print("t0 detected: : ", t_zeros)
                    break

    return t_zeros

def compute_height_window(t_zeros, l, m, ftd_s, ftd_e):
    top_mean_windows=[]
        
    # print("found peaks in: ", t_zeros)
    if len(t_zeros)==0:
        print("Warning: No peaks present")
    else:  
        after_t=l+ftd_s
        before_end=l+m-ftd_e
        for t in t_zeros:
            window=[t+after_t, t+before_end]
            top_mean_windows.append(window)
            # print("window in : ", window)
    return top_mean_windows

def compute_int_window(t_zeros, pre_delay, width):
    tp_int_windows=[]
        
    # print("found peaks in: ", t_zeros)
    if len(t_zeros)==0:
        print("Warning: No peaks present")
    else:  
        after_t = -pre_delay+width
        for t in t_zeros:
            window=[t-pre_delay, t+after_t]
            tp_int_windows.append(window)
            # print("window in : ", window)
    return tp_int_windows