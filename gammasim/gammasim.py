import numpy as np
import random
import json
import exp
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from typing import Union
import plot_utils
from configuration_parser import ConfigModel
import time
from typing import Optional

class GammaSim:
    def __init__(self, configfile_path, seed=30) -> None:
        """
        Create an object GammaSim, a simulator for GAMMA-FLASH data from a configuration file.
        ## Args
        * `configfile_path`: configuration file
        """
        random.seed(seed)
        np.random.seed(seed)
        
        with open(configfile_path, 'r') as configfile:
            self._cfg = ConfigModel(**json.load(configfile))
        
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
        tmp_lookup_tb = np.cumsum(self.__m_list)
        self.__lookup_table = np.append([0], tmp_lookup_tb)
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

    """
    def __generate_tstart(self, sampling_time):
        # Initialize t_start with zeros
        self.__t_start = np.zeros(self.__total_size, dtype=np.int64)
        # Define the possible choices as a 2D array (each row is a possible choice for an event)
        choices = np.tile(np.arange(self._cfg.tstart_min, self._cfg.tstart_max, dtype=np.int16), (self.__total_size, 1))
        
        # Loop for each peak
        for i in range(max(self.__m_list)):
            # Get indices of the i-th peak for each event
            idxs_peak_ith = self.__lookup_table[:-1] + i
            idxs_peak_ith = idxs_peak_ith[idxs_peak_ith < self.__lookup_table[1:]]
            
            # For each index, generate a t_start that respects delta_tstart
            for idx in idxs_peak_ith:
                # Create a mask to exclude choices within delta_tstart of already chosen t_start values
                mask = (choices[idx] < self.__t_start[idx] - self._cfg.delta_tstart) | \
                       (choices[idx] > self.__t_start[idx] + self._cfg.delta_tstart)
                
                # Apply the mask to get only valid choices, ensuring valid_choices is an array
                valid_choices = np.atleast_1d(choices[idx][mask])
                
                # Check if there are valid choices available
                if len(valid_choices) > 0:
                    # Randomly select a valid choice for t_start
                    self.__t_start[idx] = np.random.choice(valid_choices, 1, replace=False)[0]
                else:
                    # Raise an error if no valid choices remain
                    raise ValueError(f"Non ci sono scelte valide per `t_start` per l'evento con indice {idx}.")
            # Generate t_start for i-th peaks by randomly selecting from valid choices
            #self.__t_start[idxs_peak_ith] = [np.random.choice(vc, 1, replace=False)[0] for vc in valid_choices if len(vc) > 0]
    """
    
    def __generate_tstart(self):
        # Initialize t_start with zeros
        self.__t_start = np.zeros(self.__total_size, dtype=np.int64)
        # Define the possible choices for each curve
        choices = np.arange(self._cfg.tstart_min, self._cfg.tstart_max, dtype=np.int16)
        # Define the mask for select valid choices, thanks to probability distribution to pass
        #   to the method np.random.choice
        mask = np.full((self._cfg.size, self._cfg.tstart_max-self._cfg.tstart_min), True)
        # Loop for max number of peaks times
        for i in range(max(self.__m_list)):
            # Get the indexes for all curves which have More than i+1 Peaks
            idxs_MiPeaks = np.where(self.__m_list >= i+1)[0]
            # Get t_start idxs for the i-th peak of each curve which have More than i Peaks
            idxs_tstart  = self.__lookup_table[idxs_MiPeaks] + i
            # Re-compute the mask for the current peak 
            if i > 0:
                mask[idxs_MiPeaks] = mask[idxs_MiPeaks-1] & (\
                            (choices[None, :] < self.__t_start[idxs_tstart -1, None] - self._cfg.delta_tstart) | \
                            (choices[None, :] > self.__t_start[idxs_tstart -1, None] + self._cfg.delta_tstart)
                        )
            # Get the probability distribution from the mask
            p_distr = mask/np.sum(mask, axis=1)[:, None]
            # For each index, generate a t_start that respects self._cfg.delta_tstart
            for j, k in zip(idxs_tstart, idxs_MiPeaks):
                # Randomly select a valid choice for t_start
                self.__t_start[j] = np.random.choice(choices, 1, p=p_distr[k])[0] 

    def __generate_params(self):
        self.__x_base = self._cfg.bkgbase_level * np.ones_like(self.__t)
        self.__gamma            = np.random.choice(a=self.peak_values_poss, 
                                                   size=(self.__total_size,),
                                                   p=self.peak_value_distr)
        if self._cfg.wf_shape == 1:
            self.__shape_method = exp.apply_exp_tau
            self.__time         = self.__d
            self.__baseline     = np.zeros(self._cfg.xlen)
            self.__dt           = 1
            self.__tau1         = np.random.randint(self._cfg.tau1_min, self._cfg.tau1_max, size=self.__total_size)
            self.__tau2         = np.random.randint(self._cfg.tau2_min, self._cfg.tau2_max, size=self.__total_size)
            self.__gauss_ker    = np.full(self.__total_size, None)
            self.__gauss_ker_dt = np.full(self.__total_size, None)
            self.__p            = np.full(self.__total_size, None)
        elif self._cfg.wf_shape == 2:
            self.__shape_method = exp.second_ord_exp_decay
            self.__time         = self.__t
            self.__baseline     = np.zeros(self._cfg.xlen)
            self.__dt           = self._cfg.sampling_time
            self.__tau1         = np.random.uniform(self._cfg.tau1_min, self._cfg.tau1_max, size=(self.__total_size,))
            self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self.__total_size,))
            self.__gauss_ker    = np.random.uniform(self._cfg.gauss_kernel_min, self._cfg.gauss_kernel_max, size=(self.__total_size,))
            self.__gauss_ker_dt = self.__gauss_ker * self._cfg.sampling_time
            self.__p            = np.full(self.__total_size, None)
        elif self._cfg.wf_shape == 3:
            self.__shape_method = exp.first_ord_exp_decay
            self.__time         = self.__t
            self.__baseline     = np.zeros(self._cfg.xlen)
            self.__dt           = self._cfg.sampling_time
            self.__tau1         = np.full(self.__total_size, None)
            self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self.__total_size,))
            self.__gauss_ker    = np.full(self.__total_size, None)
            self.__gauss_ker_dt = np.full(self.__total_size, None)
            self.__p            = np.full(self.__total_size, None)
        # TODO: da aggiungere metodo 4
        elif self._cfg.wf_shape == 4:
            self.__shape_method = exp.orsa_pulse_fitting
            self.__time         = self.__t
            self.__baseline     = 0.0
            self.__dt           = self._cfg.sampling_time
            self.__tau1         = np.random.uniform(self._cfg.tau1_min, self._cfg.tau1_max, size=(self.__total_size,))
            self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self.__total_size,))
            self.__gauss_ker    = np.full(self.__total_size, None)
            self.__gauss_ker_dt = np.full(self.__total_size, None)
            self.__p            = np.random.uniform(self._cfg.p_min, self._cfg.p_max, size=(self.__total_size,))
        # self.__generate_tstart(self.__dt)
        self.__generate_tstart()
        self.__reorder_t_start()
        
    ##########################################################################################################################
    ### 3. GENERATE CURVES FOR EACH PEAK
    def __generate_peaksignal(self):
        # Generate the peak signals with the specified shape method
        self.__peak_signals = np.zeros((self.__total_size, 
                                        self._cfg.xlen))
        self.__heights = np.zeros(self.__total_size)
        for i in tqdm(range(self.__total_size)):
            self.__peak_signals[i] = self.__shape_method(self.__time, 
                                                         self.__baseline, 
                                                         self.__t_start[i], 
                                                         self.__gamma[i], 
                                                         self.__tau1[i], 
                                                         self.__tau2[i], 
                                                         self.__gauss_ker_dt[i],
                                                         self.__p[i])
            # Compute signals' height
            x_max = find_peaks(self.__peak_signals[i])[0][0]
            self.__heights[i] = self.__peak_signals[i][x_max]
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
                   'height': self.__heights[i], 
                   'gamma': self.__gamma[i], 
                   'tau1': self.__tau1[i],
                   'tau2': self.__tau2[i],
                   'g_kernel': self.__gauss_ker[i]} for i in range(start, stop)]
        return params

    def __get_quantity(self, quantity):
        ret = np.zeros((self._cfg.size, self._cfg.max_peaks))
        for i, start, stop in zip(range(self._cfg.size),
                                  self.__lookup_table[:-1], 
                                  self.__lookup_table[1:]):
            ret[i, 0:stop-start+1] = quantity[start:stop]
        return ret
    
    def get_dataset(self):
        return self.__dataset

    def get_labels(self):
        return self.__labels
    
    def get_labelsSplit(self):
        return self.__peak_signals
    
    def get_gammas(self):
        return self.__get_quantity(self.__gamma)
    
    def get_heights(self):
        return self.__get_quantity(self.__heights)
    
    
    def get_areas(self):
        areas = np.zeros((self._cfg.size, self._cfg.max_peaks))
        for i, start, stop in zip(range(self._cfg.size),
                                  self.__lookup_table[:-1], 
                                  self.__lookup_table[1:]):
            areas[i, 0:stop-start+1] = self.__integrals[start:stop]
        return areas


    def get_params(self):
        return [self.__params(i) for i in range(self._cfg.size)]
    
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
        axs[0].set_ylabel("counts")
        axs[1].set_ylabel("counts")
        axs[0].set_xlabel(f"time - bin {self._cfg.sampling_time} sec")
        axs[1].set_xlabel(f"time - bin {self._cfg.sampling_time} sec")
        maxima=find_peaks(self.__labels[idx], prominence=10)
        
        if len(maxima[0]) == len(self.__params(idx)):
            for m, p in zip(maxima[0], self.__params(idx)):
                axs[1].t_bar(x=m*self._cfg.sampling_time, ymin=0, ymax=p['height'], segment_length=self._cfg.sampling_time*150, color='tab:blue')
        else:
            print(maxima)
        # Fill between non-zero values with 50% transparency
        nonzero_values_labels = self.__labels[idx] >= 1
        #print(nonzero_values_labels)
        axs[1].fill_between(self.__t, 0, max(self.__labels[idx]), where=nonzero_values_labels, color='tab:red', alpha=0.35)
        # Calculate length of filled areas along x-axis
        length_fill = np.sum(nonzero_values_labels)
        # Add label showing length of filled area along x-axis
        axs[1].text(0.75, 0.95, f'Length: {length_fill}', transform=axs[1].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
        # plt.tight_layout()
        plt.show()

    def plot_gamma_distribution(self, num_bins=10):
        """
        Plot the distribution of gamma values in self.__gamma.
        
        Parameters:
        num_bins (int): Number of bins (bars) to use in the histogram. Default is 10.
        """
        # Calculate the histogram of gamma values with the specified number of bins
        gamma_counts, bin_edges = np.histogram(self.__gamma, bins=num_bins)
        
        # Calculate the center of each bin for plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Plotting with centered ticks
        plt.figure(figsize=(10, 5))
        plt.bar(bin_centers, gamma_counts, width=(bin_edges[1] - bin_edges[0]), color='skyblue', edgecolor='black')
        plt.xlabel("Gamma Values")
        plt.ylabel("Frequency")
        plt.title("Distribution of Gamma Values")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def plot_npeaks_distribution(self):
        """
        Plot the distribution of npeaks values.
        """
        # Calculate the histogram of gamma values with the specified number of bins
        npeaks_counts, bin_edges = np.histogram(self.__m_list, bins=self._cfg.max_peaks)
        
        # Calculate the center of each bin for plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Plotting with centered ticks
        plt.figure(figsize=(10, 5))
        plt.bar(bin_centers, npeaks_counts, width=(bin_edges[1] - bin_edges[0]), color='skyblue', edgecolor='black')
        plt.xlabel("npeaks")
        plt.ylabel("Frequency")
        plt.title("Distribution of npeaks Values")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    
    ##########################################################################################################################
    ##########################################################################################################################
    
    def generate_dataset(self, 
                         F_saturation: bool=False, 
                         F_random_npeaks: bool = False, 
                         peak_value_distr: Optional[np.ndarray[np.float64]]=None) -> None:
        """
        Generates a dataset of peak signals and corresponding labels, with options for peak distribution and saturation.
    
        Parameters:
            F_saturation (bool): Specifies whether to use gamma with saturation or not.
            F_random_npeaks (bool): Specifies whether to generate a random number of peaks per sample.
            peak_value_distr (np.ndarray, optional): Distribution of peak values. If None, a uniform distribution is used.
        """
        # Ensures that if F_random_npeaks is selected, the peak value distribution (peak_value_distr) must be None
        assert (not F_random_npeaks) or (peak_value_distr is None)
        
        # Set the gamma range based on the saturation flag
        gamma_min, gamma_max = (self._cfg.gamma_min_wtSat, self._cfg.gamma_max_wtSat) if F_saturation else (self._cfg.gamma_min_noSat, self._cfg.gamma_max_noSat)
        
        # Define possible peak values as a sequence between gamma_min and gamma_max
        self.peak_values_poss = np.linspace(gamma_min, gamma_max, num=gamma_max - gamma_min+1)
        # If a peak value distribution is provided, use it; otherwise, default to a uniform distribution
        if peak_value_distr is None:
            # Set a uniform distribution across possible peak values
            self.peak_value_distr = np.ones_like(self.peak_values_poss) / len(self.peak_values_poss)
        else:
            self.peak_value_distr = peak_value_distr
        # Assert that if a peak value distribution is provided, its length must match the length of peak_values_poss
        assert (self.peak_value_distr is None) or (len(self.peak_value_distr) == len(self.peak_values_poss))
        
        total_start_time = time.time()
        
        # Step 1: Generate the number of peaks for each sample
        start_time = time.time()
        print("STEP 1: Generating the number of peaks for each sample")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_mlist(F_random_npeaks)
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")
        
        # Step 2: Generate parameters for each peak
        start_time = time.time()
        print("STEP 2: Generating parameters for each peak")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_params()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")
    
        # Step 3: Generate signals for each peak
        start_time = time.time()
        print("STEP 3: Generating signals for each peak")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_peaksignal()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")
    
        # Step 4: Generate labels for each sample
        start_time = time.time()
        print("STEP 4: Generating labels")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_labels()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")        
    
        # Step 5: Apply noise to the dataset
        start_time = time.time()
        print("STEP 5: Applying noise to dataset")
        print('start_time:', f"{start_time:.6f}")
        self.__generate_dataset_noise()
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")   
        
        # Print total execution time for generating the dataset
        total_stop_time = time.time()
        print(f"TOTAL TIME FOR GENERATE DATASET = {total_stop_time-total_start_time:.8f}\n")
        print(self.__labels.shape)