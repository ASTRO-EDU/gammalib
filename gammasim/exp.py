import numpy as np

def apply_exp(t,x,t_start,gamma,a):
    y = np.concatenate([np.zeros(t_start),np.floor(gamma*np.exp(a*(t[t_start:]-t_start)))])
    return x + y

def apply_exp_tau(t, x, t_start, gamma, tau1, tau2):
    a1 = -(np.log(1e-4)/tau1)
    a2 = np.log(.01)/tau2

    x_leftzeros  = np.zeros(t_start-tau1)
    x_prepeak    = np.floor(gamma*np.exp(a1 * (t[t_start-tau1:t_start]-(t_start))))
    x_postpeak   = np.floor(gamma*np.exp(a2 * (t[t_start:t_start+tau2]-(t_start))))
    x_rightzeros = np.zeros(max(len(x)-t_start-tau2, 0))
    y = np.concatenate([x_leftzeros, x_prepeak, x_postpeak,x_rightzeros])
    return x + y

def apply_gauss_round(x, mean, dev):
    x_gauss = np.round(np.random.normal(mean, dev, size=x.shape)) + x
    x_gauss[x_gauss > 8192] = 8192
    x_gauss[x_gauss < 0] = 0
    return x_gauss