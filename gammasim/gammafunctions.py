import numpy as np
import json
import sys
sys.path.append('/home/gamma/workspace/gammalib')
import expfuncs.exp as exp
from typing import Union
import math

from pydantic import Field
from typing import Union
from typing_extensions import Annotated

from gammafunc import GammaFunc
from conf_parser import CommonConfigModel

##########################################################################################################################
# METHOD 1: Piecewise Exponential
##########################################################################################################################
# Configuration class for a single exponential signal
class CnfgPiecewiseExp(CommonConfigModel):
    method_name: Annotated[str, Field(default="Piecewise Exponential")]  # Name of the signal shape
    t1_min: Annotated[Union[int, float], Field(ge=0)]  # Minimum time to reach the peak
    t1_max: Annotated[Union[int, float], Field(ge=0)]  # Maximum time to reach the peak
    t2_min: Annotated[Union[int, float], Field(ge=0)]  # Minimum time to reach the baseline
    t2_max: Annotated[Union[int, float], Field(ge=0)]  # Maximum time to reach the baseline

# Class for the single exponential signal model
class GmPiecewiseExp(GammaFunc):
    # Method to load configuration from a JSON file
    def _load_config(self, config_path: str) -> CommonConfigModel:
        with open(config_path, 'r') as configfile:
            self._cfg = CnfgPiecewiseExp(**json.load(configfile))  # Load and parse the JSON into the configuration model
        return self._cfg

    # Method to generate the parameters of the signal
    def _generate_params(self):
        self.__time     = self._d  # Use the data (e.g., time points) from the parent class
        self.__baseline = np.zeros(self._cfg.xlen)  # Initialize the baseline for current peaks to zeros
        # Generate random times for reaching the peak and the baseline
        self.__t1 = np.random.randint(self._cfg.t1_min, self._cfg.t1_max, size=self._total_size)
        self.__t2 = np.random.randint(self._cfg.t2_min, self._cfg.t2_max, size=self._total_size)
        super()._generate_params()  # Call the parent class's method to complete parameter generation

    # Method to retrieve arguments for signal generation
    def _get_args(self, i):
        args = {
            'time': self.__time,  # Time points array
            'baseline': self.__baseline,  # Baseline array
            't1': self.__t1[i],  # Time to peak for the current sample
            't2': self.__t2[i]   # Time to baseline for the current sample
        }
        return args

    # Method to generate the signal shape based on given parameters
    def _shape_method(self, t_start, gamma, **kwargs):
        time, baseline, t1, t2 = kwargs['time'], kwargs['baseline'], kwargs['t1'], kwargs['t2']
        return exp.piecewise_exp(time, baseline, t_start, gamma, t1, t2)

##########################################################################################################################
# METHOD 2: Convolution
##########################################################################################################################
class CnfgConvolution(CommonConfigModel):
    method_name: Annotated[str, Field(default="Convolution")]  # Forma del segnale 
    tau2_min: Annotated[Union[int, float], Field(ge=0)] # Minimo tempo per raggiungere il picco
    tau2_max: Annotated[Union[int, float], Field(ge=0)] # Massimo tempo per raggiungere il picco
    taudiff_min: Annotated[Union[int, float], Field(ge=0)] # Minimo tempo per raggiungere il livello di fondo
    taudiff_max: Annotated[Union[int, float], Field(ge=0)] # Massimo tempo per raggiungere il livello di fondo
    gauss_kernel_min: Annotated[float, Field(ge=0)]  # Kernel Gaussiano minimo
    gauss_kernel_max: Annotated[float, Field(ge=0)]  # Kernel Gaussiano massimo

# Class for the Convolution model
class GmConvolution(GammaFunc):
    # Method to load configuration from a JSON file
    def _load_config(self, config_path: str) -> CommonConfigModel:
        with open(config_path, 'r') as configfile:
            self._cfg = CnfgConvolution(**json.load(configfile))  # Load and parse the JSON into the configuration model
        return self._cfg
            
    # Method to generate the parameters of the signal
    def _generate_params(self):
        self.__time         = self._t
        self.__baseline     = np.zeros(self._cfg.xlen)  # Initialize the baseline to zeros
        self._dt            = self._cfg.sampling_time
        # Generate random times constant 
        self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self._total_size,))
        self.__taudiff      = np.random.uniform(self._cfg.taudiff_min, self._cfg.taudiff_max, size=(self._total_size,))
        self.__tau1         = self.__tau2 + self.__taudiff
        self.__gauss_ker    = np.random.uniform(self._cfg.gauss_kernel_min, self._cfg.gauss_kernel_max, size=(self._total_size,))
        self.__gauss_ker_dt = self.__gauss_ker * self._cfg.sampling_time
        super()._generate_params()
    
    # Method to retrieve arguments for signal generation
    def _get_args(self, i):
        args = {
            'time': self.__time,  # Time points array
            'baseline': self.__baseline,  # Baseline array
            'tau1': self.__tau1[i],
            'tau2': self.__tau2[i],
            'sigma': self.__gauss_ker[i]}
        return args
    
    # Method to generate the signal shape based on given parameters
    def _shape_method(self, t_start, gamma, **kwargs):
        time, baseline, tau1, tau2, sigma = kwargs['time'], kwargs['baseline'], kwargs['tau1'], kwargs['tau2'], kwargs['sigma']
        return exp.conv_decay(time, baseline, t_start, gamma, tau1, tau2, sigma)
    
##########################################################################################################################
# METHOD 3: Convolution first order
##########################################################################################################################
# Configuration class for a First Order Convolution signal
class CnfgConvolutionFOrd(CommonConfigModel):
    method_name: Annotated[str, Field(default="First Order  Convolution")]  # Forma del segnale 
    tau_min: Annotated[Union[int, float], Field(ge=0)] # Minimo tempo per raggiungere il picco
    tau_max: Annotated[Union[int, float], Field(ge=0)] # Massimo tempo per raggiungere il picco

# Class for the First Order Convolution signal model
class GmConvolutionFOrd(GammaFunc):
    def _load_config(self, config_path: str) -> CommonConfigModel:
        with open(config_path, 'r') as configfile:
            self._cfg = CnfgConvolutionFOrd(**json.load(configfile))
        return self._cfg
            
    # Method to generate the parameters of the signal
    def _generate_params(self):
        self.__time         = self._t
        self.__baseline     = 0.0
        self._dt            = self._cfg.sampling_time
        self.__tau          = np.random.uniform(self._cfg.tau_min, self._cfg.tau_max, size=(self._total_size,))
        super()._generate_params()
    
    # Method to retrieve arguments for signal generation
    def _get_args(self, i):
        args = {
            'time': self.__time,  # Time points array
            'baseline': self.__baseline,  # Baseline array
            'tau': self.__tau[i]
            }
        return args
    
    # Method to generate the signal shape based on given parameters
    def _shape_method(self, t_start, gamma, **kwargs):
        time, baseline, tau = kwargs['time'], kwargs['baseline'], kwargs['tau']
        return exp.single_exp(time, baseline, t_start, gamma, tau)

##########################################################################################################################
# METHOD 4: ORSA
##########################################################################################################################
# Configuration class for a ORSA signal
class CnfgORSA(CommonConfigModel):
    method_name: Annotated[str, Field(default="ORSA")]  # Forma del segnale 
    tau1_min: Annotated[Union[int, float], Field(ge=0)] # Minimo tempo per raggiungere il picco
    tau1_max: Annotated[Union[int, float], Field(ge=0)] # Massimo tempo per raggiungere il picco
    tau2_min: Annotated[Union[int, float], Field(ge=0)] # Minimo tempo per raggiungere il picco
    tau2_max: Annotated[Union[int, float], Field(ge=0)] # Massimo tempo per raggiungere il picco
    p_min: Annotated[float, Field(ge=0)]  # p minimo
    p_max: Annotated[float, Field(ge=0)]  # p massimo

class GmORSA(GammaFunc):
    # Method to load configuration from a JSON file
    def _load_config(self, config_path: str) -> CommonConfigModel:
        with open(config_path, 'r') as configfile:
            self._cfg = CnfgORSA(**json.load(configfile))
        return self._cfg
            
    # Method to generate the parameters of the signal
    def _generate_params(self):
        self.__time         = self._t
        self.__baseline     = np.zeros(self._cfg.xlen)
        self._dt            = self._cfg.sampling_time
        self.__tau1         = np.random.uniform(self._cfg.tau1_min, self._cfg.tau1_max, size=(self._total_size,))
        self.__tau2         = np.random.uniform(self._cfg.tau2_min, self._cfg.tau2_max, size=(self._total_size,))
        self.__p            = np.random.uniform(self._cfg.p_min, self._cfg.p_max, size=(self._total_size,))
        super()._generate_params()
    
    # Method to retrieve arguments for signal generation
    def _get_args(self, i):
        args = {
            'time': self.__time,  # Time points array
            'baseline': self.__baseline,  # Baseline array
            'tau1': self.__tau1[i],
            'tau2': self.__tau2[i],
            'p': self.__p[i]
        }
        return args
    
    # Method to generate the signal shape based on given parameters
    def _shape_method(self, t_start, gamma, **kwargs):
        time, baseline, tau1, tau2, p = kwargs['time'], kwargs['baseline'], kwargs['tau1'], kwargs['tau2'], kwargs['p']
        return exp.orsa_pulse_fitting(time, baseline, t_start, gamma, tau1, tau2, p)

##########################################################################################################################
# METHOD 5: Semi-Gaussian
##########################################################################################################################
# Configuration class for a Semi-Gaussian CR-RC^n signal
class CnfgSemiGaussian(CommonConfigModel):
    method_name: Annotated[str, Field(default="Semi-Gaussian CR-RC^n")]  # Forma del segnale 
    tau_min: Annotated[Union[int, float], Field(ge=0)] # Minimo tempo per raggiungere il picco
    tau_max: Annotated[Union[int, float], Field(ge=0)] # Massimo tempo per raggiungere il picco
    n_min: Annotated[float, Field(ge=0)]  # n minimo
    n_max: Annotated[float, Field(ge=0)]  # n massimo

# Class for the Semi-Gaussian CR-RC^n signal model
class GmSemiGaussian(GammaFunc):
    def _load_config(self, config_path: str) -> CommonConfigModel:
        with open(config_path, 'r') as configfile:
            self._cfg = CnfgSemiGaussian(**json.load(configfile))
        return self._cfg
            
    # Method to generate the parameters of the signal
    def _generate_params(self):
        self.__time         = self._t
        self.__baseline     = 0.0
        self._dt            = self._cfg.sampling_time
        self.__tau          = np.random.uniform(self._cfg.tau_min, self._cfg.tau_max, size=(self._total_size,))
        self.__n            = np.random.randint(self._cfg.n_min, self._cfg.n_max, size=(self._total_size,))
        super()._generate_params()
    
    # Method to retrieve arguments for signal generation
    def _get_args(self, i):
        args = {
            'time': self.__time,  # Time points array
            'baseline': self.__baseline,  # Baseline array
            'tau': self.__tau[i],
            'n': self.__n[i]
        }
        return args
    
    # Method to generate the signal shape based on given parameters
    def _shape_method(self, t_start, gamma, **kwargs):
        time, baseline, tau, n = kwargs['time'], kwargs['baseline'], kwargs['tau'], kwargs['n']
        return exp.semigaussian_shaper(time, baseline, t_start, gamma, tau, n)