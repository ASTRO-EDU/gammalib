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
    

def quantize_signal(input_signal, n_q=14 , input_min=-1, input_max=1):


    q_signal = round(input_signal)
    return q_signal

def assign_interval(n, n_q=14):
    round_n = round(n)
    for i in range(n_q):
        if round_n >= 2**i and round_n < 2**(i+1):
            return 2**i
        

def apply_gauss_round(x, mean, dev):
    x_gauss = np.round(np.random.normal(mean, dev, size=x.shape)) + x
    x_gauss[x_gauss > 8192] = 8192
    x_gauss[x_gauss < 0] = 0
    return x_gauss