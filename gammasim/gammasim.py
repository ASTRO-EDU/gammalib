import numpy as np
import json
import exp
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from typing import Union

class GammaSim:
    def __init__(self, configfile_path) -> None:
        """
        Create an object GammaSim, a simulator for GAMMA-FLASH data from a configuration file.
        ## Args
        * `configfile_path`: configuration file
        """
        with open(configfile_path, 'r') as configfile:
            config = json.load(configfile)
        # Imposta dinamicamente gli attributi dall'oggetto JSON
        for key, value in config.items():
            setattr(self, key, value)
        
        self.__t = np.arange(0, self.xlen)

        if self.gauss_std == "none":
            self.gauss_std = self.gauss_maxrate * self.maxcount_value
        else:
            self.gauss_std = float(self.gauss_std)
        
        if self.gauss_std == "none":
            self.gauss_mean = self.gauss_maxrate * self.maxcount_value
        else:
            self.gauss_mean = float(self.gauss_std)

    def __generate_tstarts(self, npeaks):
        t_starts = [] 
        while len(t_starts) < npeaks:  
            # Generate a candidate t_start within the specified range
            t_start_candidate = np.random.randint(self.tstart_min, self.tstart_max)  
            # Check if the candidate is at least delta_tstart away from all existing t_starts
            if all(abs(t_start_candidate - ts) >= self.delta_tstart for ts in t_starts):
                t_starts.append(t_start_candidate) 
        return t_starts


    def generate_dataset(self, F_saturation: bool) -> None:
        """
        Generate the dataset
        ## Args
        * F_saturation:
        """
        self.__params       = []
        self.__dataset      = np.empty((self.size, self.xlen), dtype=np.int16)
        self.__labels_split = np.empty((self.size, self.max_peaks*self.xlen))
        self.__labels       = np.empty((self.size, self.xlen))
        self.__integrals    = np.empty((self.size, self.max_peaks))
        # Background of the signal
        x_base = self.bkgbase_level * np.ones_like(self.__t)
        # Select gamma_min and gamma_max depending on the saturation flag
        gamma_min, gamma_max = (self.gamma_min_wtSat, self.gamma_max_wtSat) if F_saturation else (self.gamma_min_noSat, self.gamma_max_noSat)
        for i in tqdm(range(len(self.__dataset))):
            peak_params        = []
            peak_signals       = []
            peak_integrals     = []
            # Generate the tstart in a safe mode from overlaps
            tstarts = self.__generate_tstarts(self.max_peaks)
            for j in range(self.max_peaks):
                # Generate the parameters
                gamma   = np.random.randint(gamma_min, gamma_max) 
                tau1    = np.random.randint(self.tau1_min, self.tau1_max)
                tau2    = np.random.randint(self.tau2_min, self.tau2_max)
                t_start = tstarts[j]
                # Create the peak signal
                peak_signal  = exp.apply_exp_tau(self.__t, 
                                                 np.zeros_like(self.__t), 
                                                 t_start, gamma, tau1, tau2)
                # Save metadata
                peak_params.append({'t_start': t_start, 'gamma': gamma, 'tau1': tau1, 'tau2': tau2})
                peak_signals.append(peak_signal)
                peak_integrals.append(np.sum(peak_signal))
            # Sort peak_params, peak_signals, and peak_integrals based on t_start
            sorted_data = sorted(zip(peak_params, peak_signals, peak_integrals), key=lambda x: x[0]['t_start'])
            peak_params, peak_signals, peak_integrals = zip(*sorted_data)
            # Save current peak parameters
            self.__params.append(peak_params)
            # Add the background to the peak signal
            x_total = x_base + np.sum(peak_signals, axis=0)
            # Apply Gauss noise 
            x_total_noise = exp.apply_gauss_round(x_total, self.gauss_mean, self.gauss_std)
            # integrals
            self.__integrals[i][:self.max_peaks] = peak_integrals
            # Append dataset
            self.__dataset[i] = x_total_noise
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
                idx = np.random.randint(0, self.size)
        else:
            raise Exception(f"Type for {idx} not allowed")
        param_string = '\n'.join([str(param) for param in self.__params[idx]])
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].set_title(param_string)
        axs[0].step(self.__t, self.__dataset[idx], color='tab:blue')
        axs[1].set_title(f'wf_{idx:05d}, \nAreas: {self.__integrals[idx]}')
        axs[1].step(self.__t, self.__labels[idx], color='tab:red')
        # Fill between non-zero values with 50% transparency
        nonzero_values_labels = self.__labels[idx] != 0
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