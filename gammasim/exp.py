import numpy as np
from math import sqrt
from scipy.special import erf

def apply_exp(t,x,t_start,gamma,a):
    y = np.concatenate([np.zeros(t_start),np.floor(gamma*np.exp(a*(t[t_start:]-t_start)))])
    return x + y

def apply_exp_tau(t, x, t_start, gamma, tau1, tau2, sigma):
    a1 = -(np.log(1/gamma)/tau1)
    a2 = np.log(1/gamma)/tau2

    x_leftzeros  = np.zeros(t_start-tau1)
    x_prepeak    = gamma*np.exp(a1 * (t[t_start-tau1:t_start]-(t_start)))
    x_postpeak   = gamma*np.exp(a2 * (t[t_start:t_start+tau2]-(t_start)))
    x_rightzeros = np.zeros(max(len(x)-t_start-tau2, 0))
    y = np.concatenate([x_leftzeros, x_prepeak, x_postpeak,x_rightzeros])
    return x + y
    

def _erf_term(t, t_start, sigma, tau):
    return erf((t-t_start)/(sqrt(2)*sigma)-(sqrt(2)*sigma)/(2*tau))

def _single_exp_fn(t, t_start, sigma, tau):
    return np.exp(((sigma**2)/(2*tau**2))-(t-t_start)/tau)

def second_ord_exp_decay(t, x, t_start, gamma, tau1, tau2, sigma):
    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    x_leftzeros  = np.zeros_like(t_left)
    t_right = t[split_index:]  # All values from t_start onward
    
    x_resp = gamma*(_single_exp_fn(t_right, t_start, sigma, tau1)*(1+_erf_term(t_right, t_start, sigma, tau1))-
                     _single_exp_fn(t_right, t_start, sigma, tau2)*(1+_erf_term(t_right, t_start, sigma, tau2)))

    y = np.concatenate([x_leftzeros, x_resp])
    return x + y

def first_ord_exp_decay(t, x, t_start, gamma, tau1, tau2, sigma):
    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    x_leftzeros  = np.zeros_like(t_left)
    t_right = t[split_index:]  # All values from t_start onward
    
    x_resp = gamma*(_single_exp_fn(t_right, t_start, 0, tau2))

    y = np.concatenate([x_leftzeros, x_resp])
    return x + y
    
def quantize_signal(input_signal, n_bit, input_min, input_max):
    # Initialize the output array with zeros
    output_s = np.zeros_like(input_signal)
    
    n_q = 2**n_bit
    # Calculate the quantization step size
    q_interval = input_max - input_min
    step_size = q_interval / (n_q - 1)
    
    # Generate quantization levels (q_values)
    q_values = np.linspace(input_min, input_max, n_q)
    
    # Perform quantization for each value in the input signal
    for i in range(len(input_signal)):
        if input_signal[i] <= input_min:
            output_s[i] = q_values[0]
        elif input_signal[i] >= input_max:
            output_s[i] = q_values[-1]
        else:
            # Find the closest quantization level
            diff = np.abs(q_values - input_signal[i])
            output_s[i] = q_values[np.argmin(diff)]
    
    return output_s

        

def apply_gauss(x, mean, dev):
    x_gauss = np.random.normal(mean, dev, size=x.shape) + x
    return x_gauss