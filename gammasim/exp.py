import numpy as np
from math import sqrt, factorial
from scipy.special import erf
import math

def apply_exp(t, x, t_start, gamma, a):
    y = np.concatenate([np.zeros(t_start),np.floor(gamma*np.exp(a*(t[t_start:]-t_start)))])
    return x + y
    
def _erf_term(t, t_start, sigma, tau):
    if sigma==0:
        return np.zeros_like(t)
    return erf((t-t_start)/(sqrt(2)*sigma)-(sqrt(2)*sigma)/(2*tau))

def _single_exp_fn(t, t_start, sigma, tau, n=None):
    return np.exp(((sigma**2)/(2*tau**2))-(t-t_start)/tau)

#####################################################################################################################################################################

##############
# METHOD 1
##############

def apply_exp_tau(t, x, t_start, gamma, tau1, tau2, sigma, p, n):
    eps = 1    
    a1 = -(math.log(1/gamma)/tau1)
    a2 = math.log(1/gamma)/tau2

    x_leftzeros  = np.zeros(t_start-tau1)
    x_prepeak    = gamma*np.exp(a1 * (t[t_start-tau1:t_start]-(t_start)))
    x_postpeak   = gamma*np.exp(a2 * (t[t_start:t_start+tau2]-(t_start)))
    x_rightzeros = np.zeros(max(len(x)-t_start-tau2, 0))
    y = np.concatenate([x_leftzeros, x_prepeak, x_postpeak,x_rightzeros])
    return x + y

##############
# METHOD 2
##############

def second_ord_exp_decay(t, x, t_start, gamma, tau1, tau2, sigma, p, n):
    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    x_leftzeros  = np.zeros_like(t_left)
    t_right = t[split_index:]  # All values from t_start onward
    
    x_resp = gamma*(_single_exp_fn(t_right, t_start, sigma, tau1)*(1+_erf_term(t_right, t_start, sigma, tau1))-
                     _single_exp_fn(t_right, t_start, sigma, tau2)*(1+_erf_term(t_right, t_start, sigma, tau2)))
    y = np.concatenate([x_leftzeros, np.abs(x_resp)])
    return x + y

##############
# METHOD 3
##############

def first_ord_exp_decay(t, x, t_start, gamma, tau1, tau2, sigma, p, n):
    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    x_leftzeros  = np.zeros_like(t_left)
    t_right = t[split_index:]  # All values from t_start onward
    
    x_resp = gamma*(_single_exp_fn(t_right, t_start, 0, tau2))

    y = np.concatenate([x_leftzeros, x_resp])
    return x + y

##############
# METHOD 4
##############

def orsa_pulse_fitting(t, y0, t_start, gamma, tau1, tau2, sigma, p, n):
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

##############
# METHOD 5
##############

def semigaussian_shaper(t, y0, t_start, gamma, tau1, tau2, sigma, p, n):
    """
    # METHOD 5
    ## SEMI-GAUSSIAN SHAPER Digital shaping function for semi-Gaussian filters.
    ref: https://ieeexplore.ieee.org/abstract/document/6026241

    ### Parameters:
    * t       : numpy array, sampled times (assumed to be ordered)
    * y0      : float, baseline value
    * t_start : float, signal start time
    * gamma   : float, signal amplitude
    * tau1    : float, growth time constant (not used in this function)
    * tau2    : float, decay time constant
    * sigma   : float, standard deviation of noise (not used in this function)
    * p       : float, parameter for shaping (not used in this function)
    * n       : int, order of the semi-Gaussian shaping

    ### Returns:
    * y : numpy array, shaped signal at each time `t`
    """
    y = np.where(
        t < t_start,  # For t < t_start, the signal remains at the baseline value y0
        y0,
        y0 + gamma * (1.0 / factorial(n)) * (((t - t_start) / tau2) ** n) * np.exp(-(t - t_start) / tau2)  # For t >= t_start
    )
    return y

#####################################################################################################################################################################
    
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