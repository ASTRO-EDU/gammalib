import numpy as np

def apply_exp(t,x,t_start,gamma,a):
    y = np.concatenate([np.zeros(t_start),np.floor(gamma*np.exp(a*(t[t_start:]-t_start)))])
    return x + y

def apply_exp_tau(t, x, t_start, gamma, tau1, tau2):

    split_index = np.where(t >= t_start)[0][0]
    t_left = t[:split_index]  # All values before t_start
    print(t_left.shape)
    x_leftzeros  = np.zeros_like(t_left)
    print(x_leftzeros.shape)
    t_right = t[split_index:]  # All values from t_start onward
    x_resp = gamma*(np.exp(-(t_right-t_start)/tau1)-np.exp(-(t_right-t_start)/tau2))

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