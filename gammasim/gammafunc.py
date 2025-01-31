from abc import ABC, abstractmethod
import numpy as np
import random
import sys
sys.path.append('/home/gamma/workspace/gammalib')
import expfuncs.exp as exp
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from typing import Union
import plot_utils
import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union
from typing_extensions import Annotated

from conf_parser import CommonConfigModel

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
    

class GammaFunc(ABC):
    @abstractmethod
    def _load_config(self, config_path: str)-> CommonConfigModel:
        """
        Load configuration file from config file path
        """
        # with open(configfile_path, 'r') as configfile:
        #     self._cfg = ConfigModel(**json.load(configfile))
        pass
    
    def __init__(self, configfile_path, seed=30) -> None:
        """
        Create an object GammaSim, a simulator for GAMMA-FLASH data from a configuration file.
        ## Args
        * `configfile_path`: configuration file
        """
        random.seed(seed)
        np.random.seed(seed)
        self._cfg = self._load_config(configfile_path)
        
        # Set self attributes based on config fields
        self._d = np.arange(0, self._cfg.xlen, dtype=np.int16)
        self._dt = 1
        self._t = self._d * self._cfg.sampling_time
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
        self._lookup_table = np.append([0], tmp_lookup_tb)
        self._total_size = self._lookup_table[-1]

    ##########################################################################################################################
    ### 2. GENERATE PARAMETERS FOR EACH PEAK
    def __reorder_t_start(self):
        # Crea una copia di t_start per evitare di modificare direttamente l'array originale
        reordered_t_start = np.zeros_like(self._t_start)
        # Cicla sui sottoinsiemi definiti da lookup_table
        for i in range(self._cfg.size):
            # Ottieni gli indici di inizio e fine del sottoinsieme i-esimo
            start_idx = self._lookup_table[i]
            end_idx = self._lookup_table[i + 1]
            # Prendi il sottoinsieme corrispondente di t_start e lo ordina
            reordered_t_start[start_idx:end_idx] = np.sort(self._t_start[start_idx:end_idx])
        # Array t_start riordinato
        self._t_start = reordered_t_start * self._dt
    
    def __generate_tstart(self):
        # Initialize t_start with zeros
        self._t_start = np.zeros(self._total_size, dtype=np.int64)
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
            idxs_tstart  = self._lookup_table[idxs_MiPeaks] + i
            # Re-compute the mask for the current peak 
            if i > 0:
                mask[idxs_MiPeaks] = mask[idxs_MiPeaks-1] & (\
                            (choices[None, :] < self._t_start[idxs_tstart -1, None] - self._cfg.delta_tstart) | \
                            (choices[None, :] > self._t_start[idxs_tstart -1, None] + self._cfg.delta_tstart)
                        )
            # Get the probability distribution from the mask
            p_distr = mask/np.sum(mask, axis=1)[:, None]
            # For each index, generate a t_start that respects self._cfg.delta_tstart
            for j, k in zip(idxs_tstart, idxs_MiPeaks):
                # Randomly select a valid choice for t_start
                self._t_start[j] = np.random.choice(choices, 1, p=p_distr[k])[0] 

    @abstractmethod
    def _generate_params(self):
        self.__x_base = self._cfg.bkgbase_level * np.ones_like(self._t)
        self._gamma   = np.random.choice(a=self.peak_values_poss, 
                                         size=(self._total_size,),
                                         p=self.peak_value_distr)
        self.__generate_tstart()
        self.__reorder_t_start()
      
    ##########################################################################################################################
    ### 3. GENERATE CURVES FOR EACH PEAK
    @abstractmethod
    def _get_args(self, i) -> dict:
        pass

    @abstractmethod
    def _shape_method(self, **args):
        pass

    def __generate_peaksignal(self):
        # Generate the peak signals with the specified shape method
        self.__peak_signals = np.zeros((self._total_size, 
                                        self._cfg.xlen))
        self._heights = np.zeros(self._total_size)
        for i in tqdm(range(self._total_size)):
            self.__peak_signals[i] = self._shape_method(self._t_start[i], 
                                                        self._gamma[i], 
                                                        **self._get_args(i))
            # Compute signals' height
            x_max = np.argmax(self.__peak_signals[i])
            #x_max = find_peaks(self.__peak_signals[i])[0][0]
            self._heights[i] = self.__peak_signals[i][x_max]
        # Compute signals' area 
        self.__integrals = np.sum(self.__peak_signals, axis=1)
        
    ##########################################################################################################################
    ### 4. COMPOSE DATASET TO HAVE LABELS 
    def __generate_labels(self):
        self.__labels = np.array(
                            [np.sum(
                                self.__peak_signals[
                                    self._lookup_table[i]:self._lookup_table[i+1], :
                            ], axis=0) for i in range(self._cfg.size)])

    ##########################################################################################################################
    ### 5. APPLY GAUSS NOISE  
    def __generate_dataset_noise(self, F_saturation):
        # Apply Gauss noise 
        labels_noise = exp.apply_gauss(self.__labels + self.__x_base[None, :], 
                                       self._cfg.gauss_mean, self._cfg.gauss_std)
        maxcount_value = np.rint(np.max(labels_noise))
        if F_saturation:
            maxcount_value = self._cfg.maxcount_value
        # Apply quantization
        self.__dataset = np.array(
            [exp.quantize_signal(labels_noise[i], 
                                 self._cfg.n_bit_quantization, 
                                 self._cfg.mincount_value, 
                                 maxcount_value) for i in range(self._cfg.size)], 
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
                start_idx = self._lookup_table[i]
                end_idx = self._lookup_table[i+1]
                # Fill the row with the corresponding integrals, adding zeros if needed
                reshaped_integrals[i, :end_idx-start_idx] = self.__integrals[start_idx:end_idx]
        return reshaped_integrals

#   @abstractmethod
    def _params(self, idx_sample:int=0):
        # {'t_start': t_start, 'height': height, 'gamma': gamma, 'tau1': tau1, 'tau2': tau2, 'g_kernel': gauss_ker}
        start = self._lookup_table[idx_sample]
        stop = self._lookup_table[idx_sample + 1]
        params = []
        for i in range(start, stop):
            args = self._get_args(i)
            args.update(
                {'t_start': self._t_start[i], 
                'height': self._heights[i], 
                'gamma': self._gamma[i]})
            args.pop("time")
            args.pop("baseline")
            params.append(args)
        return params
        

    def __get_quantity(self, quantity):
        if len(quantity.shape) > 1:
            ret = np.zeros((self._cfg.size, self._cfg.max_peaks, quantity.shape[-1]))
        else:
            ret = np.zeros((self._cfg.size, self._cfg.max_peaks))
        for i, start, stop in zip(range(self._cfg.size),
                                  self._lookup_table[:-1], 
                                  self._lookup_table[1:]):
            ret[i, 0:stop-start+1] = quantity[start:stop]
        return ret
    
    def get_dataset(self):
        return self.__dataset

    def get_labels(self):
        return self.__labels
    
    def get_labelsSplit(self):
        return self.__get_quantity(self.__peak_signals)
        return self.__peak_signals
    
    def get_gammas(self):
        return self.__get_quantity(self._gamma)
    
    def get_heights(self):
        return self.__get_quantity(self._heights)
    
    
    def get_areas(self):
        return self.__reshape_integrals()

    def get_params(self):
        return [self._params(i) for i in range(self._cfg.size)]
    
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
            ', '.join(
                f"{k}: "
                f"{f'{v:.2f}' if isinstance(v, float) and v > 1 else f'{v/1e-9:.1f}e-9' if isinstance(v, float) and v < 1 else f'{v:.{decimal_places}f}'.rstrip('0').rstrip('.') if isinstance(v, (int, float)) else v}"
                for k, v in param.items() if v is not None
            )
            for param in params
        )

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].set_title(print_params(self._params(idx)), fontsize=8)  # Imposta la dimensione del carattere a 10
        axs[0].step(self._t, self.__dataset[idx], color='tab:blue')
        axs[1].set_title(f'wf_{idx:05d}, \nAreas: {np.round(self.__reshape_integrals()[idx], 2)}')
        axs[1].plot(self._t, self.__labels[idx], color='tab:red')
        axs[0].set_ylabel("counts")
        axs[1].set_ylabel("counts")
        axs[0].set_xlabel(f"time - bin {self._cfg.sampling_time} sec")
        axs[1].set_xlabel(f"time - bin {self._cfg.sampling_time} sec")
        maxima=find_peaks(self.__labels[idx], prominence=10)
        
        if len(maxima[0]) == len(self._params(idx)):
            for m, p in zip(maxima[0], self._params(idx)):
                axs[1].t_bar(x=m*self._cfg.sampling_time, ymin=0, ymax=p['height'], segment_length=self._cfg.sampling_time*150, color='tab:blue')
        else:
            print(maxima)
        # Fill between non-zero values with 50% transparency
        nonzero_values_labels = self.__labels[idx] >= 1
        #print(nonzero_values_labels)
        axs[1].fill_between(self._t, 0, max(self.__labels[idx]), where=nonzero_values_labels, color='tab:red', alpha=0.35)
        # Calculate length of filled areas along x-axis
        length_fill = np.sum(nonzero_values_labels)
        # Add label showing length of filled area along x-axis
        axs[1].text(0.75, 0.95, f'Length: {length_fill}', transform=axs[1].transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
        # plt.tight_layout()
        plt.show()

    def plot_gamma_distribution(self, num_bins=10):
        """
        Plot the distribution of gamma values in self._gamma.
        
        Parameters:
        num_bins (int): Number of bins (bars) to use in the histogram. Default is 10.
        """
        # Calculate the histogram of gamma values with the specified number of bins
        gamma_counts, bin_edges = np.histogram(self._gamma, bins=num_bins)
        
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

    def plot_areas_distribution(self, num_bins=10):
        """
        Plot the distribution of gamma values in self._gamma.
        
        Parameters:
        num_bins (int): Number of bins (bars) to use in the histogram. Default is 10.
        """
        # Calculate the histogram of gamma values with the specified number of bins
        gamma_counts, bin_edges = np.histogram(self.__integrals, bins=num_bins)
        
        # Calculate the center of each bin for plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Plotting with centered ticks
        plt.figure(figsize=(10, 5))
        plt.bar(bin_centers, gamma_counts, width=(bin_edges[1] - bin_edges[0]), color='skyblue', edgecolor='black')
        plt.xlabel("Area Values")
        plt.ylabel("Frequency")
        plt.title("Distribution of Area Values")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def plot_heights_distribution(self, num_bins=10):
        """
        Plot the distribution of gamma values in self._gamma.
        
        Parameters:
        num_bins (int): Number of bins (bars) to use in the histogram. Default is 10.
        """
        # Calculate the histogram of gamma values with the specified number of bins
        gamma_counts, bin_edges = np.histogram(self._heights, bins=num_bins)
        
        # Calculate the center of each bin for plotting
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Plotting with centered ticks
        plt.figure(figsize=(10, 5))
        plt.bar(bin_centers, gamma_counts, width=(bin_edges[1] - bin_edges[0]), color='skyblue', edgecolor='black')
        plt.xlabel("Heights Values")
        plt.ylabel("Frequency")
        plt.title("Distribution of Heights Values")
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
        # gamma_min, gamma_max = (self._cfg.gamma_min_wtSat, self._cfg.gamma_max_wtSat) if F_saturation else (self._cfg.gamma_min_noSat, self._cfg.gamma_max_noSat)
        gamma_min, gamma_max = self._cfg.gamma_min, self._cfg.gamma_max
        
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
        self._generate_params()
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
        self.__generate_dataset_noise(F_saturation)
        stop_time = time.time()
        print('stop_time: ', f"{stop_time:.6f}, total time for this step = {stop_time-start_time:.8f}\n")   
        
        # Print total execution time for generating the dataset
        total_stop_time = time.time()
        print(f"TOTAL TIME FOR GENERATE DATASET = {total_stop_time-total_start_time:.8f}\n")
        print(self.__labels.shape)