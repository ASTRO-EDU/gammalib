import numpy as np
import json
import exp
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from typing import Union
import plot_utils
from configuration_parser import ConfigModel
import time

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
        # Set self attributes based on config fields
        self.__d = np.arange(0, self._cfg.xlen, dtype=np.int16)
        self.__t = self.__d * self._cfg.sampling_time
        if self._cfg.gauss_std is None:
            self._cfg.gauss_std = self._cfg.gauss_maxrate * self._cfg.maxcount_value
        if self._cfg.gauss_mean is None:
            self._cfg.gauss_mean = self._cfg.gauss_maxrate * self._cfg.maxcount_value
        # Set all the simulator attributes
        self.__dataset            = None
        self.__labels             = None
        self.__integrals          = None
        self.__reshaped_integrals = None

    ##########################################################################################################################
    ### 1. GENERATE THE NUMBER OF PEAKS FOR EACH CURVE
    def __generate_mlist(self, F_random_npeaks: bool = False):
        """
        Generate self.__m_list based on the value of F_random_npeaks.
        
        Parameters:
        F_random_npeaks (bool): If True, self.__m_list is generated as an array of random integers 
                                between 1 and self.max_peaks, with size self.size.
                                If False, self.__m_list is an array of size self.size, where each 
                                element is self.max_peaks.
    
        self.__m_list: Array generated based on the above condition.
        """
        if F_random_npeaks:
            # Generate an array of random integers between 1 and self.max_peaks
            self.__m_list = np.random.randint(1, self._cfg.max_peaks + 1, size=self._cfg.size)
        else:
            # Create an array of size self.size where each element is self.max_peaks
            self.__m_list = np.full(self._cfg.size, self._cfg.max_peaks)
        self.__lookup_table = np.append([0], np.cumsum(self.__m_list))
        self.__total_size = self.__lookup_table[-1]

    ##########################################################################################################################
    ### 2. GENERATE PARAMETERS FOR EACH PEAK
    def __reorder_t_start(self):
        # Crea una copia di t_start per evitare di modificare direttamente l'array originale
        reordered_t_start = np.zeros_like(self.__t_start)
        # Cicla sui sottoinsiemi definiti da lookup_table
        for i in range(self._cfg.size):
            # Ottieni gli indici di inizio e fine del sottoinsieme i-esimo
            start_idx = self.__lookup_table[i]
            end_idx = self.__lookup_table[i + 1]
            # Prendi il sottoinsieme corrispondente di t_start e lo ordina
            reordered_t_start[start_idx:end_idx] = np.sort(self.__t_start[start_idx:end_idx])
        # Array t_start riordinato
        self.__t_start = reordered_t_start * self.__dt
    
    def __generate_tstart(self, sampling_time):
        # Initialize t_start with zeros
        self.__t_start = np.zeros(self.__total_size, dtype=np.int64)
        # Define the possible choices as a 2D array (each row is a possible choice for an event)
        choices = np.tile(np.arange(self._cfg.tstart_min, self._cfg.tstart_max, dtype=np.int16), (self.__total_size, 1))
        # Loop reduced only for maximum number of peaks
        for i in range(max(self.__m_list)):
            # Get the i-th peak indices
            idxs_peak_ith = self.__lookup_table[:-1] + i
            idxs_peak_ith = idxs_peak_ith[idxs_peak_ith < self.__lookup_table[1:]]
            # Generate filters for choices based on delta_tstart
            filter_low = self.__t_start[idxs_peak_ith] - self._cfg.delta_tstart
            filter_high = self.__t_start[idxs_peak_ith] + self._cfg.delta_tstart
            # Apply filters on all choices in parallel with broadcasting
            mask = (choices[idxs_peak_ith] < filter_low[:, None]) | (choices[idxs_peak_ith] > filter_high[:, None])
            # Keep only valid choices for each row
            valid_choices = [c[mask_row] for c, mask_row in zip(choices[idxs_peak_ith], mask)]
            # Generate t_start for i-th peaks by randomly selecting from valid choices
            self.__t_start[idxs_peak_ith] = [np.random.choice(vc, 1, replace=False)[0] for vc in valid_choices if len(vc) > 0]

    def __generate_params(self, F_saturation:bool=False):
        # Get the right range for gamma 
        gamma_min, gamma_max = (self._cfg.gamma_min_wtSat, self._cfg.gamma_max_wtSat) if F_saturation else (self._cfg.gamma_min_noSat, self._cfg.gamma_max_noSat)
        self.__x_base = self._cfg.bkgbase_level * np.ones_like(self.__t)
        
        if self._cfg.wf_shape == 1:
            self.__shape_method = exp.apply_exp_tau
            self.__time         = self.__d
            self.__dt           = 1
            self.__tau1         = np.random.randint(self._cfg.tau1_min, self._cfg.tau1_max, size=self.__total_size)
            self.__tau2         = np.random.randint(self._cfg.tau2_min, self._cfg.tau2_max, size=self.__total_size)
            self.__gamma        = np.random.randint(gamma_min, gamma_max, size=(self.__total_size,)) 
            self.__gauss_ker    = np.full(self.__total_size, None)
            self.__gauss_ker_dt = np.full(self.__total_size, None)
        elif self._cfg.wf_shape == 2:
            self.__shape_method = exp.second_ord_exp_decay
            self.__time         = self.__t
            self.__dt           = self._cfg.sampling_time
            self.__tau1         = np.random.uniform(self._cfg.tau1_min, self._cfg.tau1_max, size=(self.__total_size,))
            self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self.__total_size,))
            self.__gamma        = np.random.randint(gamma_min, gamma_max, size=(self.__total_size,)) 
            self.__gauss_ker    = np.random.uniform(self._cfg.gauss_kernel_min, self._cfg.gauss_kernel_max, size=(self.__total_size,))
            self.__gauss_ker_dt = self.__gauss_ker * self._cfg.sampling_time
        elif self._cfg.wf_shape == 3:
            self.__shape_method = exp.first_ord_exp_decay
            self.__time         = self.__t
            self.__dt           = self._cfg.sampling_time
            self.__tau1         = np.full(self.__total_size, None)
            self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self.__total_size,))
            self.__gamma        = np.random.randint(gamma_min, gamma_max, size=(self.__total_size,)) 
            self.__gauss_ker    = np.full(self.__total_size, None)
            self.__gauss_ker_dt = np.full(self.__total_size, None)
        self.__generate_tstart(self.__dt)
        self.__reorder_t_start()
        
    ##########################################################################################################################
    ### 3. GENERATE CURVES FOR EACH PEAK
    def __generate_peaksignal(self):
        # Generate the peak signals with the specified shape method
        self.__peak_signals = np.zeros((self.__total_size, 
                                        self._cfg.xlen))
        self.__height = np.zeros(self.__total_size)
        for i in tqdm(range(self.__total_size)):
            self.__peak_signals[i] = self.__shape_method(self.__time, 
                                                         np.zeros(self._cfg.xlen), 
                                                         self.__t_start[i], 
                                                         self.__gamma[i], 
                                                         self.__tau1[i], 
                                                         self.__tau2[i], 
                                                         self.__gauss_ker_dt[i])
            # Compute signals' height
            x_max = find_peaks(self.__peak_signals[i])[0][0]
            self.__height[i] = self.__peak_signals[i][x_max]
        # Compute signals' area 
        self.__integrals = np.sum(self.__peak_signals, axis=1)
        
    ##########################################################################################################################
    ### 4. COMPOSE DATASET TO HAVE LABELS 
    def __generate_labels(self):
        self.__labels = np.array(
                            [np.sum(
                                self.__peak_signals[
                                    self.__lookup_table[i]:self.__lookup_table[i+1], :
                            ], axis=0) for i in range(self._cfg.size)])

    ##########################################################################################################################
    ### 5. APPLY GAUSS NOISE  
    def __generate_dataset_noise(self):
        # Apply Gauss noise 
        labels_noise = exp.apply_gauss(self.__labels + self.__x_base[None, :], 
                                       self._cfg.gauss_mean, self._cfg.gauss_std)
        # Apply quantization
        self.__dataset = np.array(
            [exp.quantize_signal(labels_noise[i], 
                                 self._cfg.n_bit_quantization, 
                                 self._cfg.mincount_value, 
                                 self._cfg.maxcount_value) for i in range(self._cfg.size)], 
            dtype=np.int16)
    
    ##########################################################################################################################
    ##########################################################################################################################

    def __reshape_integrals(self):
        if self.__reshaped_integrals == None:
            # Initialize an array of zeros with the desired shape
            reshaped_integrals = np.zeros((self._cfg.size, self._cfg.max_peaks))
            # Loop over each subset defined by lookup_table and fill the reshaped array
            for i in range(self._cfg.size):
                # Calculate the start and end indices for the current subset in integrals
                start_idx = self.__lookup_table[i]
                end_idx = self.__lookup_table[i+1]
                # Fill the row with the corresponding integrals, adding zeros if needed
                reshaped_integrals[i, :end_idx-start_idx] = self.__integrals[start_idx:end_idx]
        return reshaped_integrals

    def __params(self, idx_sample:int=0):
        # {'t_start': t_start, 'height': height, 'gamma': gamma, 'tau1': tau1, 'tau2': tau2, 'g_kernel': gauss_ker}
        start = self.__lookup_table[idx_sample]
        stop = self.__lookup_table[idx_sample + 1]
        params = [{'t_start': self.__t_start[i], 
                   'height': self.__height[i], 
                   'gamma': self.__gamma[i], 
                   'tau1': self.__tau1[i],
                   'tau2': self.__tau2[i],
                   'g_kernel': self.__gauss_ker[i]} for i in range(start, stop)]
        return params

    def get_dataset(self):
        return self.__dataset

    def get_labels(self):
        return self.__labels
    
    def get_labelsSplit(self):
        return self.__peak_signals
    
    def get_areas(self):
        return self.__integrals

    def get_params(self):
        return [self.__params(i) for i in range(self.__total_size)]
    
    def get_sampling_time(self):
        return self._cfg.sampling_time
    
    def get_shape_method(self):
        return self._cfg.wf_shape
    
    ##########################################################################################################################
    ##########################################################################################################################
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
                idx = np.argmax(np.sum(self.__reshape_integrals(), axis=1))
            elif idx == "min":
                idx = np.argmin(np.sum(self.__reshape_integrals(), axis=1))
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
        axs[0].set_title(print_params(self.__params(idx)), fontsize=8)  # Imposta la dimensione del carattere a 10
        axs[0].step(self.__t, self.__dataset[idx], color='tab:blue')
        axs[1].set_title(f'wf_{idx:05d}, \nAreas: {np.round(self.__reshape_integrals()[idx], 2)}')
        axs[1].plot(self.__t, self.__labels[idx], color='tab:red')
        maxima=find_peaks(self.__labels[idx], prominence=10)
        
        if len(maxima[0]) == len(self.__params(idx)):
            for m, p in zip(maxima[0], self.__params(idx)):
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

    ##########################################################################################################################
    ##########################################################################################################################
    
    def generate_dataset(self, F_saturation: bool, F_random_npeaks: bool = False) -> None:
        total_start_time = time.time()
        # Step 1:
        start_time = time.time()
        print("STEP 1: number of peaks for each sample generation")
        print('start_time:', f"{start_time:.6f}")
        # print('delta_tstart', self._cfg.delta_tstart)
        self.__generate_mlist(F_random_npeaks)
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")
        
        # Step 2:
        start_time = time.time()
        print("STEP 2: parameters generation")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_params(F_saturation)
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")

        # Step 3:
        start_time = time.time()
        print("STEP 3: peaks' signals generation")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_peaksignal()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")

        # Step 4:
        start_time = time.time()
        print("STEP 4: labels generation")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_labels()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")        

        # Step 5:
        start_time = time.time()
        print("STEP 5: applying noise to dataset")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_dataset_noise()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")   
        
        total_stop_time = time.time()
        print(f"TOTAL TIME FOR GENERATE DATASET = {total_stop_time-total_start_time:.8f}\n")
        print(self.__labels.shape)