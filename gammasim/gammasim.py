import numpy as np
import json
import exp
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from typing import Union
import plot_utils
from configuration_parser import ConfigModel

class GammaSim:
    def __init__(self, configfile_path) -> None:
        """
        Create an object GammaSim, a simulator for GAMMA-FLASH data from a configuration file.
        ## Args
        * `configfile_path`: configuration file
        """
        with open(configfile_path, 'r') as configfile:
            
            self._cfg = ConfigModel(**json.load(configfile))
            print(self._cfg)
        # Imposta gli attributi di self in base ai campi di config

        self.__d = np.arange(0, self._cfg.xlen)
        self.__t = self.__d*self._cfg.sampling_time

        if self._cfg.gauss_std is None:
            self._cfg.gauss_std = self._cfg.gauss_maxrate * self._cfg.maxcount_value
        
        if self._cfg.gauss_mean is None:
            self._cfg.gauss_mean = self._cfg.gauss_maxrate * self._cfg.maxcount_value

    def __generate_tstarts(self, npeaks, sampling_time):
        t_starts = [] 
        while len(t_starts) < npeaks:  
            # Generate a candidate t_start within the specified range
            t_start_candidate = np.random.randint(self._cfg.tstart_min, self._cfg.tstart_max)*sampling_time
            # Check if the candidate is at least delta_tstart away from all existing t_starts
            if all(abs(t_start_candidate - ts) >= self._cfg.delta_tstart*sampling_time for ts in t_starts):
                t_starts.append(t_start_candidate) 
        return t_starts


    def generate_dataset(self, F_saturation: bool, F_random_npeaks: bool = False) -> None:
        """
        Generate the dataset
        ## Args
        * F_saturation:
        """
        self.__params       = []
        self.__dataset      = np.empty((self._cfg.size, self._cfg.xlen), dtype=np.int16)
        self.__labels_split = np.empty((self._cfg.size, self._cfg.max_peaks*self._cfg.xlen))
        self.__labels       = np.empty((self._cfg.size, self._cfg.xlen))
        self.__integrals    = np.zeros((self._cfg.size, self._cfg.max_peaks))
        # Background of the signal
        x_base = self._cfg.bkgbase_level * np.ones_like(self.__t)
        # Select gamma_min and gamma_max depending on the saturation flag
        gamma_min, gamma_max = (self._cfg.gamma_min_wtSat, self._cfg.gamma_max_wtSat) if F_saturation else (self._cfg.gamma_min_noSat, self._cfg.gamma_max_noSat)
        
        gauss_ker = np.random.uniform(self._cfg.gauss_kernel_min, self._cfg.gauss_kernel_max)
        gauss_ker_dt = gauss_ker*self._cfg.sampling_time
        if self._cfg.wf_shape == 1:
            shape_method = exp.apply_exp_tau
            time = self.__d
            dt = 1
            tau1    = np.random.randint(self._cfg.tau1_min, self._cfg.tau1_max)
            tau2    = np.random.randint(self._cfg.tau2_min, self._cfg.tau2_max)
            gauss_ker = None
            gauss_ker_dt = None
        elif self._cfg.wf_shape == 2:
            shape_method = exp.second_ord_exp_decay
            time = self.__t
            dt=self._cfg.sampling_time
            tau1    = np.random.uniform(self._cfg.tau1_min, self._cfg.tau1_max)
            tau2    = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max)
        elif self._cfg.wf_shape == 3:
            shape_method = exp.first_ord_exp_decay
            time = self.__t
            dt=self._cfg.sampling_time
            tau1    = None
            tau2    = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max)
            gauss_ker = None
            gauss_ker_dt = None

        for i in tqdm(range(len(self.__dataset))):
            peak_params        = []
            peak_signals       = []
            peak_integrals     = []
            m = int(np.random.uniform(1, self._cfg.max_peaks, 1)) if F_random_npeaks else self._cfg.max_peaks
            # Generate the tstart in a safe mode from overlaps
            tstarts = self.__generate_tstarts(m, dt)
            # Generate the parameters
            
            for j in range(m):
                gamma   = np.random.randint(gamma_min, gamma_max) 
                # Generate the parameters
                
                t_start = tstarts[j]
                # Create the peak signal
                
                peak_signal  = shape_method(time, np.zeros_like(time), t_start, gamma, tau1, tau2, gauss_ker_dt)
                x_max = find_peaks(peak_signal)[0][0]
                height = peak_signal[x_max]
                # Save metadata
                peak_params.append({'t_start': t_start, 'height': height, 'gamma': gamma, 'tau1': tau1, 'tau2': tau2, 'g_kernel': gauss_ker })
                peak_signals.append(peak_signal)
                peak_integrals.append(np.sum(peak_signal)*self._cfg.sampling_time)
            # Sort peak_params, peak_signals, and peak_integrals based on t_start
            sorted_data = sorted(zip(peak_params, peak_signals, peak_integrals), key=lambda x: x[0]['t_start'])
            peak_params, peak_signals, peak_integrals = zip(*sorted_data)
            # Save current peak parameters
            self.__params.append(peak_params)
            # Add the background to the peak signal
            x_total = x_base + np.sum(peak_signals, axis=0)
            # Apply Gauss noise 
            x_total_noise = exp.apply_gauss(x_total, self._cfg.gauss_mean, self._cfg.gauss_std)
            # integrals
            self.__integrals[i][:m] = peak_integrals
            # apply quantization 
            x_quantized = exp.quantize_signal(x_total_noise, self._cfg.n_bit_quantization, self._cfg.mincount_value, self._cfg.maxcount_value)
            # Append dataset
            self.__dataset[i] = x_quantized
            # Combine labels
            # NOTE: TO TAKE UNDER CONTROL MEMORY CONS. COMMENTS THESE LINES
            self.__labels[i] = np.sum(peak_signals, axis=0)
            peak_signals = np.concatenate(peak_signals)
            self.__labels_split[i, :len(peak_signals)] = peak_signals

    def plot_wf(self, idx: Union[int, str] ='random') -> None:
        """
        Plot a single waveform from the dataset with its generation parameters:
        ## __Args__
        * `idx`: if idx is `int` it will print `idx`-th element of the dataset. When it is a `str` type
          it can assume the following values:
            - `max`: it plots the waveform with the maximum area
            - `min`: it plots the waveform with the minimum area
            - `random`: it plots a random waveform
        """
        # Extract idx
        if type(idx) == int:
            pass
        elif type(idx) == str:
            if idx == "max":
                idx = np.argmax(np.sum(self.__integrals, axis=1))
            elif idx == "min":
                idx = np.argmin(np.sum(self.__integrals, axis=1))
            elif idx == "random":
                idx = np.random.randint(0, self._cfg.size)
        else:
            raise Exception(f"Type for {idx} not allowed")
        
        print_params = lambda params, decimal_places=10: '\n'.join(
            ', '.join(f"{k}: {f'{v:.{decimal_places}f}'.rstrip('0').rstrip('.') if isinstance(v, (int, float)) else v}"
                    for k, v in param.items() if v is not None) 
            for param in params
        )   

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].set_title(print_params(self.__params[idx]), fontsize=8)  # Imposta la dimensione del carattere a 10
        axs[0].step(self.__t, self.__dataset[idx], color='tab:blue')
        axs[1].set_title(f'wf_{idx:05d}, \nAreas: {self.__integrals[idx]}')
        axs[1].plot(self.__t, self.__labels[idx], color='tab:red')
        maxima=find_peaks(self.__labels[idx], prominence=10)
        
        if len(maxima[0]) == len(self.__params[idx]):
            for m, p in zip(maxima[0], self.__params[idx]):
                axs[1].t_bar(x=m*self._cfg.sampling_time, ymin=0, ymax=p['height'], segment_length=self._cfg.sampling_time*150, color='tab:blue')
        else:
            print(maxima)
        # Fill between non-zero values with 50% transparency
        nonzero_values_labels = self.__labels[idx] >= 1
        print(nonzero_values_labels)
        axs[1].fill_between(self.__t, 0, max(self.__labels[idx]), where=nonzero_values_labels, color='tab:red', alpha=0.35)
        # Calculate length of filled areas along x-axis
        length_fill = np.sum(nonzero_values_labels)
        # Add label showing length of filled area along x-axis
        axs[1].text(0.75, 0.95, f'Length: {length_fill}', transform=axs[1].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
        # plt.tight_layout()
        plt.show()

    def get_dataset(self):
        return self.__dataset

    def get_labels(self):
        return self.__labels
    
    def get_labelsSplit(self):
        return self.__labels_split
    
    def get_areas(self):
        return self.__integrals

    def get_params(self):
        return self.__params
    
    def get_sampling_time(self):
        return self._cfg.sampling_time
    
    def get_shape_method(self):
        return self._cfg.wf_shape