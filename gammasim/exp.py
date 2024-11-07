import numpy as np
from math import sqrt
from scipy.special import erf
import math

def apply_exp(t,x,t_start,gamma,a):
    y = np.concatenate([np.zeros(t_start),np.floor(gamma*np.exp(a*(t[t_start:]-t_start)))])
    return x + y

def apply_exp_tau(t, x, t_start, gamma, tau1, tau2, sigma, p):
    eps = 1    
    a1 = -(math.log(1/gamma)/tau1)
    a2 = math.log(1/gamma)/tau2

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

def second_ord_exp_decay(t, x, t_start, gamma, tau1, tau2, sigma, p):
    gamma = -gamma
    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    x_leftzeros  = np.zeros_like(t_left)
    t_right = t[split_index:]  # All values from t_start onward
    
    x_resp = gamma*(_single_exp_fn(t_right, t_start, sigma, tau1)*(1+_erf_term(t_right, t_start, sigma, tau1))-
                     _single_exp_fn(t_right, t_start, sigma, tau2)*(1+_erf_term(t_right, t_start, sigma, tau2)))

    y = np.concatenate([x_leftzeros, x_resp])
    return x + y

def first_ord_exp_decay(t, x, t_start, gamma, tau1, tau2, sigma, p):
    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    x_leftzeros  = np.zeros_like(t_left)
    t_right = t[split_index:]  # All values from t_start onward
    
    x_resp = gamma*(_single_exp_fn(t_right, t_start, 0, tau2))

    y = np.concatenate([x_leftzeros, x_resp])
    return x + y

def orsa_pulse_fitting(t, y0, t_start, gamma, tau1, tau2, sigma, p):
    """
    # METHOD 4
    ## ORSA fitting function for noisy signal reconstruction.
    ref: https://pubs.aip.org/aip/rsi/article-abstract/81/10/10D321/357646/Energy-resolution-of-gamma-ray-spectroscopy-of-JET?redirectedFrom=fulltext

    ### Parameters:
    * t       : array numpy, sample times (assume they are ordered)
    * y0      : float, baseline value
    * gamma   : float, signal amplitude
    * t_start : float, signal start time
    * tau1    : float, exponential growth time constant
    * tau2    : float, exponential decay time constant
    * p       : float, parameter that modulates the nonlinearity of exponential growth
    
    ### Returns:
    S_t : array numpy, valore della forma d'onda digitalizzata in corrispondenza di ciascun tempo t
    """
    y = np.where(
        t < t_start,
        y0,  # For t < t0, the signal is equal to y0
        y0 + (gamma * ((1 - np.exp(-(t - t_start) / tau1))**p)) * np.exp(-(t - t_start) / tau2)  # Per t >= t0
    )
    return y
    
def quantize_signal(input_signal, n_bit, input_min, input_max):
    # Calculate the number of quantization levels and the step size
    n_q = 2**n_bit
    step_size = (input_max - input_min) / (n_q - 1)
    
    # Clip the input signal to stay within input_min and input_max
    input_clipped = np.clip(input_signal, input_min, input_max)
    
    # Map the input signal to quantization levels
    scaled_input = (input_clipped - input_min) / step_size
    quantized_indices = np.round(scaled_input).astype(int)
    
    # Generate quantization levels
    q_values = np.linspace(input_min, input_max, n_q)
    
    # Use the quantized indices to get the quantized signal
    output_s = q_values[quantized_indices]
    
    return output_s   

def apply_gauss(x, mean, dev):
    x_gauss = np.random.normal(mean, dev, size=x.shape) + x
    return x_gauss