import os
import sys
import glob
import tables
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import re
from time import time
from tables import *
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.optimize import curve_fit 
from scipy.special import factorial 
from scipy.special import gamma as gamma_function 
from scipy.special import gammainc as lowinc_gamma_function 
from scipy.special import erf 
import scipy.stats as stats
from tables.description import Float32Col
from tables.description import Float64Col
from typing import Annotated, Literal, Union
import h5py
import ast


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
# sys.path.append('/home/gamma/workspace/gammalib')
import expfuncs.exp as expfuncs

class Eventlist_fitting:
    def __init__(self):
        self.results_df = None
        self.fName      = None
        self.fToFit     = None
        self.par_names  = None
        self.results_df = None

#####################################################################################################################################################################
    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    def find_extremes(self, array, smoothing_radius, der_radius):
        smoothing_radius = int(smoothing_radius)
        der_radius = int(der_radius)
        smooth_array = [np.divide(np.sum(array[k-smoothing_radius:k+smoothing_radius]),2*smoothing_radius) for k in np.add(range(len(array)-2*smoothing_radius),smoothing_radius)]
        ##### build derivative with fit around a kernel of radius der_fit_radius
        der_range = np.add(range(len(smooth_array)-2*der_radius),der_radius)
        array_der = []
        for fit_center in der_range:
            [m_fit, q_fit], rest = curve_fit(
                lambda x, m, q: m*x+q, 
                range(2*der_radius), smooth_array[fit_center - der_radius : fit_center + der_radius]
            )
            array_der.append(m_fit)
        #### identify max and min points
        max_points = []
        min_points = []
        for this_i in range(len(array_der)-1):
            if array_der[this_i] > 0 and array_der[this_i+1] <0:
                max_points.append(this_i + der_radius + smoothing_radius)
            elif array_der[this_i] < 0 and array_der[this_i+1] >0:
                min_points.append(this_i + der_radius + smoothing_radius)
        return smooth_array, max_points, min_points
#####################################################################################################################################################################
    def allowedFuncs(self):
        return [
            self.double_exp.__name__,
            self.convolution_func.__name__,
            self.single_exp.__name__,
            self.orsa_function.__name__,
            self.semigaussian_shaper.__name__,
            self.semigaussian_shaper_with_tail.__name__,
        ]
    #           REMOVED FUNCS:
    #           self.double_exp_nocut.__name__,

    ##############
    # METHOD 1
    ##############
    def double_exp(self, t, y0,t0, t1,t2, A, tau1,tau2): 
        return expfuncs.double_exp(t, y0, t0, A, t1, t2, tau1, tau2)

    def double_exp_nocut(self,t, y0,t1, A, tau1,tau2):
        y = np.where(t<t1,
                     y0 + A*np.exp((t-t1)/(tau1)),
                     y0 + A*np.exp((t-t1)/(-tau2)))     
        return y

    ##############
    # METHOD 2
    ##############
    def convolution_func(self, t, y0, t0, A, tau1, tau2, sigma):
        return expfuncs.conv_decay(t, y0, t0, A, tau1, tau2, sigma)

    ##############
    # METHOD 3
    ##############
    def single_exp(self, t, y0,t0, A, tau):
        return expfuncs.single_exp(t, y0, t0, A, tau)

    ##############
    # METHOD 4
    ##############
    def orsa_function(self,t, y0, t0, A, tau1, tau2, P):
        return expfuncs.orsa_pulse_fitting(t, y0, t0, A, tau1, tau2, P)
    
    ##############
    # METHOD 5
    ##############
    def semigaussian_shaper(self, t, y0, t0, n, A, tau):
        return expfuncs.semigaussian_shaper(t, y0, t0, A, tau, n)

    ##############
    # METHOD 6
    ##############
    def semigaussian_shaper_with_tail(self, t, y0, t0, n, A, tau, B, t1, sigma):
        return expfuncs.semigaussian_shaper_with_tail(t, y0, t0, A, tau, n, B, t1, sigma)


    ###########################
    #### process_file ####
    ###########################
    def __parameters_checks(self,
                            fName: Literal['double_exp', 'convolution_func', 'single_exp', 'orsa_function', 
                                           'semigaussian_shaper', 'semigaussian_shaper_with_tail'] = 'double_exp', 
                            detector: Literal['SiPM', 'PMT'] = 'SiPM',
                            mode: Literal['w', 'a'] = 'w'):
        if mode == 'a':
            if hasattr(self, 'results_df') and isinstance(self.results_df, pd.DataFrame):
                # Check if metadata matches
                if (self.results_df.attrs.get('fName') != fName or 
                    self.results_df.attrs.get('detector') != detector):
                    raise ValueError(
                        "Metadata mismatch: Existing DataFrame has different 'fName' or 'detector' attributes."
                    )
        # Check detector type 
        if detector not in ['SiPM', 'PMT']:
            print(f'Detector {detector} not found. Cannot proceed. \nAvailable detectors are PMT and SiPM')
            raise Exception(msg_exc)
        # Get saturation value for each detector
        if detector == 'SiPM':
            self.saturationValue = 1200
        else:
            self.saturationValue = 8191
        self.detector = detector
        # Check on fit function
        try:
            self.fName = fName
            self.fToFit = getattr(self, fName)
            self.__par_names = [name for name in self.fToFit.__code__.co_varnames[2:]] #### parameters of input function 
            self.par_names = [f'arg__{name}' for name in self.fToFit.__code__.co_varnames[2:]] #### parameters of input function 
        except:
            msg_exc = f'Function {fName} not found. Cannot proceed.\nFunctions available for fitting: \n {", ".join(self.allowedFuncs())}'
            raise Exception(msg_exc)
        #### END CHECKS #### 
        

    def __process_wf(self,
                     y: npt.NDArray[np.int_],
                     i_wf: int,
                     printPlot: bool = False,
                     evalSat: bool = False, 
                     log: bool = False,
                     prevSat: bool = False):
        peak_id = []
        chisquare = []
        all_pars = []
        all_covs = []
        max_curve = []
        integ = []
        dof = []
        #if evalSat:
        satCheck = []
        ### arr = PMT 0 SiPM 250
        ### arrmov = PMT 15, SiPM 3
        if self.detector == 'SiPM':
            arr = y+250 # mean1=-190 -> mean1=58 -> mean2=104.4
            arrmov = self.moving_average(arr, 3)
        elif self.detector == 'PMT':
            arr = y+0
            arrmov = self.moving_average(arr, 15)
        #
        arr3 = arr[:100]
        mmean1 = arr3.mean() #### initial bkg estimate
        stdev1 = arr3.std()  #### stdev estimate
        mmean2 = mmean1 * 2 * 0.9 
        #
        peaks, values = find_peaks(arrmov , height=mmean2, width=15, distance=25)
        #
        if log == True:
            print(f"Waveform num. {i_wf} ##############################")
            print(f"mean {mmean1} and stdev {stdev1}")
        #
        # deltav = PMT 20 SiPM 25 (actually used SiPM: 50)
        if self.detector == 'SiPM':
            deltav = 50
        elif self.detector == 'PMT':
            deltav = 20
        #
        peaks2 = np.copy(peaks)
        #filtraggio picchi
        for v in peaks2:
            arrcalcMM = arrmov[v] - arrmov[v-deltav:v+deltav]
            #80 has been chosen with heuristics on data 
            ind = np.where(arrcalcMM[:] > 80)
            #remove peaks too small or peaks too close to the end of the wf
            if len(ind[0]) == 0 or v > 16000:
                peaks = peaks[peaks != v]
        
        if not len(peaks) == 0:
            j=0
            for v in peaks:
                try:
                    ##############################################
                    #### DEFINITION OF SIGNAL ARRAY arrSignal ####
                    ##############################################
                    #calculation on raw data
                    arrcalc = arr[v-deltav:]
                    rowsL = np.where(arrcalc[:] < mmean1)[0]
                    indexRows=np.where(rowsL >= deltav)[0]
                    if v < 15000 or len(indexRows) > 0:
                        arrSignal = arrcalc[0:rowsL[indexRows[0]]]
                    else:
                        arrSignal = arrcalc
                    ###################
                    #### CHECK SAT ####
                    ###################
                    if max(arrSignal) > self.saturationValue:
                        thisSat = True
                    else:
                        thisSat = False
                    ########################
                    #### START FIT CODE ####
                    ########################
                    #                        
                    ##################################
                    #### HEURISTIC FIT PARAMETERS ####
                    ##################################
                    n_explore = 100
                    if self.fToFit.__name__ == self.semigaussian_shaper.__name__:
                        initial_guesses = [mmean1, 0.1, 16, (max(arrSignal)-mmean1)*10, 3]
                        lower_bounds = [this_guess/100 for this_guess in initial_guesses]
                        upper_bounds = [this_guess*100 for this_guess in initial_guesses]
                        lower_bounds[2] = 1
                        upper_bounds[2] = 50
                        if thisSat:
                            initial_guesses[4] = 5
                                
                    elif self.fToFit.__name__ == self.semigaussian_shaper_with_tail.__name__:
                        initial_guesses = [mmean1, 0.1, 16, (max(arrSignal)-mmean1)*10, 3,
                                            50,130, 40]
                        lower_bounds = [this_guess/100 for this_guess in initial_guesses]
                        upper_bounds = [this_guess*100 for this_guess in initial_guesses]
                        lower_bounds[2] = 1
                        upper_bounds[2] = 50
                        lower_bounds[5] = 1
                        upper_bounds[5] = 100
                        if thisSat:
                            #initial_guesses[3] = initial_guesses[3]*5
                            initial_guesses[4] = 5
                            for this_par in [5,6,7]:
                                initial_guesses[this_par] = -1e-9
                                lower_bounds[this_par] = -2e-9
                                upper_bounds[this_par] = 0
                    elif self.fToFit.__name__ == self.orsa_function.__name__:
                        initial_guesses = [mmean1, np.where(arrSignal > mmean1+5*stdev1)[0][0], max(arrSignal)-mmean1, 30, 70, 1]
                        lower_bounds = [initial_guesses[0]-n_explore*stdev1,
                                        initial_guesses[1]-n_explore*stdev1,
                                        initial_guesses[2]-n_explore*stdev1,
                                        initial_guesses[3]/100,
                                        initial_guesses[4]/100,
                                        initial_guesses[5]/100]
                        upper_bounds = [initial_guesses[0]+n_explore*stdev1,
                                        initial_guesses[1]+n_explore*stdev1,
                                        initial_guesses[2]+n_explore*stdev1,
                                        initial_guesses[3]*100,
                                        initial_guesses[4]*100,
                                        initial_guesses[5]*100]
                    elif self.fToFit.__name__ == self.single_exp.__name__:
                        initial_guesses = [mmean1, np.where(arrSignal == max(arrSignal))[0][0], max(arrSignal)-mmean1, 30]
                        lower_bounds = [initial_guesses[0]-3*stdev1,
                                        initial_guesses[1]-n_explore*stdev1,
                                        initial_guesses[2]-n_explore*stdev1,
                                        initial_guesses[3]/n_explore]
                        upper_bounds = [initial_guesses[0]+3*stdev1,
                                        initial_guesses[1]+n_explore*stdev1,
                                        initial_guesses[2]+n_explore*stdev1,
                                        initial_guesses[3]*n_explore]
                    
                    elif self.fToFit.__name__ == self.double_exp.__name__:
                        initial_guesses = [mmean1,
                                            np.where(arrSignal > mmean1 + 5*stdev1)[0][0],
                                            np.where(arrSignal == max(arrSignal))[0][0],
                                            np.where(arrSignal > mmean1 + 5*stdev1)[0][-1],
                                            max(arrSignal)-mmean1,
                                            3, 30]
                        lower_bounds = [this_guess/100 for this_guess in initial_guesses]
                        upper_bounds = [this_guess*100 for this_guess in initial_guesses]
            
                    elif self.fToFit.__name__ == self.double_exp_nocut.__name__:
                        initial_guesses = [mmean1,
                                            np.where(arrSignal == max(arrSignal))[0][0],
                                            max(arrSignal)-mmean1,
                                            3, 30]
                        lower_bounds = [this_guess/100 for this_guess in initial_guesses]
                        upper_bounds = [this_guess*100 for this_guess in initial_guesses]                
            
                    elif self.fToFit.__name__ == self.convolution_func.__name__:
                        if self.detector == 'SiPM':
                            initial_guesses = [np.abs(mmean1),
                                                np.abs(np.where(arrSignal > mmean1+5*stdev1)[0][0]),# == max(arrSignal)
                                                np.abs(max(arrSignal)-mmean1),
                                                18, 12, 1]
                        elif self.detector == 'PMT':
                            initial_guesses = [np.abs(mmean1),
                                                1,#np.abs(np.where(arrSignal > mmean1+5*stdev1)[0][0]),# == max(arrSignal)
                                                np.abs(max(arrSignal)-mmean1),
                                                18, 12, 1]
                        lower_bounds = [this_guess/100 for this_guess in initial_guesses]
                        upper_bounds = [this_guess*100 for this_guess in initial_guesses]

                    ### fixed bounds if initial guess == 0
                    for i_bound in range(len(lower_bounds)):
                        if upper_bounds[i_bound] <= lower_bounds[i_bound]:
                            upper_bounds[i_bound] = upper_bounds[i_bound] + 1e-9

                    ###################
                    ####### FIT #######
                    ###################
                    maxPosition = np.argmax(arrSignal)
                    start_fit = 0
                    if thisSat or (self.detector == 'PMT') or ((self.detector == 'SiPM') and (self.fName == 'semigaussian_shaper_with_tail') and (not prevSat)):
                        end_fit = len(arrSignal)-1
                    elif (self.detector == 'SiPM') & (prevSat):
                        end_fit = maxPosition
                        initial_guesses[0] = -50
                        lower_bounds[0] = initial_guesses[0]-n_explore*stdev1
                        upper_bounds[0] = initial_guesses[0]+n_explore*stdev1
                    else: ### Remaining cases not sat, not post sat, not semigaus with tail SiPM fit
                        ### Finds the fit end using the derivative
                        smoothing_radius = 5
                        der_fit_radius = 2
                        smooth_arrSignal, max_arrSignal, min_arrSignal = self.find_extremes(arrSignal, smoothing_radius=smoothing_radius, der_radius = der_fit_radius)
                        end_fit = 0
                        for min_place in range(len(min_arrSignal)):
                            if min_arrSignal[min_place] > maxPosition:
                                end_fit = min_arrSignal[min_place]
                                break
                        
                        if end_fit == 0:
                            end_fit = 2*maxPosition
                    arrToFit = arrSignal[start_fit:end_fit]

                    if evalSat:
                        xToFit = []
                        yToFit = []
                        for numArr, valArr in enumerate(arrToFit):
                            if valArr < self.saturationValue:
                                xToFit.append(numArr)
                                yToFit.append(valArr)
                    else:
                        xToFit = range(len(arrToFit))
                        yToFit = arrToFit 
                        
                    fit_results = curve_fit(self.fToFit, xToFit, yToFit,#range(len(arrToFit)),arrToFit,
                                            p0=initial_guesses,
                                            bounds=(lower_bounds,upper_bounds),
                                            sigma=stdev1, method='trf')
                    par_vals = fit_results[0]
                    par_covs = fit_results[1]
                    
                    residuals = [(self.fToFit(t_res, *par_vals) - arrToFit[t_res])/stdev1 for t_res in xToFit]#range(len(arrToFit))]
                    best_curve = [self.fToFit(t, *par_vals) for t in xToFit]
                
                    if log == True:
                        print(f"\tEXECUTED FIT for waveform {i_wf} on its peak number {j}") 
                    # Calculate chi2
                    dof_tmp = len(xToFit)-len(self.__par_names) #len(arrToFit)-len(par_names)
                    chisquare_fit = np.sum(np.multiply(residuals,residuals))/dof_tmp
                    # Integration range
                    start_integ_tmp = 0
                    end_integ_tmp = len(arrSignal)-1
                    arrToInteg = arrSignal[start_integ_tmp:end_integ_tmp]
                    ####
                    #### for semigaussian_shaper_with_tail, only the first gaussian is integrated
                    if (self.fName == 'semigaussian_shaper_with_tail'):
                        fNameToInteg = 'semigaussian_shaper'
                    else:
                        fNameToInteg = self.fName
                    
                    fToInteg = getattr(self, fNameToInteg)
                    npars_fToInteg = len([name for name in fToInteg.__code__.co_varnames[2:]])
                    par_vals_redux = par_vals[:npars_fToInteg]
                    curveToInteg = [fToInteg(t, *par_vals_redux) for t in range(len(arrToInteg))]

                    for t in np.add(range(len(arrToInteg)-maxPosition),maxPosition):
                        if curveToInteg[t] < par_vals[0]:
                            end_integ_tmp = t
                            break
                        
                    #end_integ_tmp = min(2*maxPosition, end_integ_tmp)
                    integ_tmp = sum(curveToInteg[start_integ_tmp:end_integ_tmp])-(end_integ_tmp-start_integ_tmp)*par_vals[0]
                    #else:
                    #    integ_tmp = sum(best_curve[start_integ_tmp:end_integ_tmp])-(end_integ_tmp-start_integ_tmp)*par_vals[0]
                    # record curve max
                    max_curve_tmp = max(curveToInteg)
                    #max_place_tmp = np.argmax(best_curve)
                    # save peak identifier
                    peak_id.append((i_wf, j))

                    ####################
                    #### APPENDING ####
                    ####################
                    all_pars.append(par_vals)
                    all_covs.append(par_covs)
                    dof.append(dof_tmp)
                    chisquare.append(chisquare_fit)
                    integ.append(integ_tmp)
                    max_curve.append(max_curve_tmp)

                    #if evalSat:
                    satCheck.append(thisSat)
                
                    if (log == True) or (printPlot == True):
                        ###################
                        #### FIT PLOT #####
                        ###################
                        plt.figure()
                        plt.plot(range(len(arrSignal)),arrSignal,color='b')
                        ##### PLOT INITIAL GUESS
                        y_plot_guess = [self.fToFit(t_plot, *initial_guesses) for t_plot in range(len(arrSignal))]
                        plt.plot(range(len(arrSignal)),y_plot_guess, color='g') # guess plot
                        ##### FIT PLOT
                        plt.plot(np.add(range(len(arrToInteg)),start_integ_tmp),curveToInteg, color='deeppink')
                        plt.plot(xToFit,best_curve, color='m') #np.add(range(len(arrToFit)),start_fit)
                        plt.plot(range(len(arrSignal)), [par_vals[0] for t in arrSignal], color='r', linestyle='--')
                        ##### HIGHLIGHT FIT and INTEG RANGE
                        if thisSat:
                            fitStarts = []
                            fitEnds = []
                            fitStarts.append(start_fit)
                            for i_thisx in range(len(xToFit)-1):
                                if xToFit[i_thisx+1] > xToFit[i_thisx] + 1:
                                    fitEnds.append(xToFit[i_thisx])
                                    fitStarts.append(xToFit[i_thisx+1])
                            fitEnds.append(end_fit)

                            for this_edge in range(len(fitStarts)):
                                plt.axvspan(fitStarts[this_edge], fitEnds[this_edge], color='g', alpha=0.2)
                        else:
                            plt.axvspan(start_fit, end_fit, color='g', alpha=0.2)
                        #plt.axvspan(start_integ_tmp, end_integ_tmp, color='orange', alpha=0.2)
                        plt.title(f'{self.fName}_{self.detector}_{i_wf}_{j}')

                        if printPlot == True:
                            plt.savefig(f'singlefit_plot/singlefit_{self.fName}_{self.detector}_{i_wf}_{j}.png', bbox_inches='tight')
                            
                        if (log == True) or prevSat or thisSat:
                            plt.show()
                        
                        ###################
                        ## END FIT PLOT ###
                        ###################
                    if thisSat:
                        prevSat = True
                    else:
                        prevSat = False
                    ###################
                    #### END FIT ######
                    ###################
                except Exception as e:
                    print(f"\tEXCEPTION: cannot fit waveform {i_wf} peak {j}")
                    ##traceback.print_exc()
                    continue
                j = j + 1
        else:            
            print(f'Block {self._blockID_counter} | Waveform {i_wf}: No peaks found.')
        return peak_id, chisquare, all_pars, all_covs, max_curve, integ, dof, satCheck

#####################################################################################################################################################################
    
    def process_arr(self, 
                    data: npt.NDArray[np.int_],
                    fName: Literal['double_exp', 'convolution_func', 'single_exp', 'orsa_function', 
                                   'semigaussian_shaper', 'semigaussian_shaper_with_tail'] = 'double_exp', 
                    detector: Literal['SiPM', 'PMT'] = 'SiPM', 
                    evalSat: bool = False, 
                    printPlot: bool = False,
                    startEvent: int = 0, 
                    endEvent: int = -1,
                    log: bool = False,
                    mode: Literal['w', 'a'] = 'w'):
        """
        Processes a dataset and extracts fit parameters for all peaks in the data.

        Parameters
        ----------
        `data` : numpy.ndarray
            A 2D array containing the dataset to process, where each row represents a waveform.
        `fName` : {'double_exp', 'convolution_func', 'single_exp', 'orsa_function', 
                'semigaussian_shaper', 'semigaussian_shaper_with_tail'}, default='double_exp'
            The name of the fitting function to use. Determines the model applied during peak fitting.
        `detector` : {'SiPM', 'PMT'}, default='SiPM'
            The type of detector associated with the dataset. Used to configure specific fitting behavior.
        `evalSat` : bool, default=False
            Whether to evaluate waveform saturation. If True, additional checks for saturation will be performed.
        `printPlot` : bool, default=False
            If True, plots of the waveforms and fits will be displayed for visual inspection.
        `startEvent` : int, default=0
            The index of the first waveform to process in the dataset.
        `endEvent` : int, default=-1
            The index of the last waveform to process in the dataset. Use -1 to process all waveforms from `startEvent`.
        `log` : bool, default=False
            If True, verbose logging will be printed to the console, including details of the fitting process.
        `mode` : {'w', 'a'}, default='w'
        Determines whether the results should overwrite the previous DataFrame ('w') 
        or append to it ('a').

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the following columns:
            - `wfID`: The waveform ID.
            - `pkID`: The peak ID within the waveform.
            - `chi2`: The chi-squared value of the fit for each peak.
            - `curveMax`: The maximum value of the fitted curve.
            - `integral`: The integral of the fitted curve.
            - `dof`: Degrees of freedom for the fit.
            - `satCheck`: A flag indicating if the waveform was saturated.
            - `arg__<param>`: The value of each fit parameter (e.g., `arg__amplitude`, `arg__sigma`, etc.).
            - `covs__<param>`: The covariance values for each fit parameter.

            The DataFrame also includes metadata attributes:
            - `fName`: The name of the fitting function used.
            - `detector`: The type of detector used.

        Notes
        -----
        - The method processes each waveform in the specified range (`startEvent` to `endEvent`) individually.
        - Each waveform is analyzed using the `__process_wf` method, which performs the actual peak detection and fitting.
        - Fit results for each waveform and its peaks are accumulated in a structured format and returned as a DataFrame.
        - This method is designed to handle large datasets efficiently, with the ability to enable or disable logging and plotting as needed.

        Raises:
        -------
        `ValueError`
            If 'mode' is 'a' and the metadata (fName or detector) do not match the existing DataFrame.

        Example
        -------
        >>> data = np.random.randint(0, 100, size=(1000, 512))  # Simulated dataset
        >>> results_df = obj.process_arr(data, 
                                        fName='double_exp', 
                                        detector='SiPM', 
                                        evalSat=True, 
                                        printPlot=False, 
                                        startEvent=0, 
                                        endEvent=100, 
                                        log=True)
        >>> print(results_df.head())
        
        """
        
        # Validate input parameters
        self.__parameters_checks(fName, detector=detector, mode=mode)

        # Initialize or update blockID counter
        if mode=='w' or (not hasattr(self, '_blockID_counter')):
            self._blockID_counter = 1
        else:
            self._blockID_counter += 1
        blockID = self._blockID_counter

        ########################
        #### INITIALIZATION ####
        ########################
        results = [] # Placeholder for results

        if log == True:
            print(f'##############################\n'
                  f'Processing with blockID={blockID} | Function: {self.fToFit.__name__} | Detector: {detector}')

        # Initialize a dictionary to store results for each waveform and peak
        results = {
            "blockID": [],   # Block array ID
            "wfID": [],      # Waveform ID
            "pkID": [],      # Peak ID
            "chi2": [],      # Chi-squared values for fits
            "curveMax": [],  # Maximum values of the fitted curves
            "integral": [],  # Integral of the curve
            "dof": [],       # Degrees of freedom in the fit
            "satCheck": [],  # Saturation status
        }
            
        # Add keys for fit parameters and their covariance matrices
        for k in self.__par_names:
            results[f'arg__{k}'] = []  # Fit parameter values
            results[f'covs__{k}'] = []  # Covariance values for each parameter

        ########################
        #### DATA SELECTION ####
        ########################
        # Restrict the processing to the specified range of events
        if endEvent==-1: 
            endEvent = len(data)
        data_section = data[startEvent:endEvent]
        prevSat = False  # Track if the previous waveform was saturated

        #############################
        #### PROCESS EACH EVENT ####
        #############################
        for y, i in zip(data_section, range(len(data_section))):
            i_wf = i + startEvent  # Current waveform index
            
            # Process the current waveform and retrieve results
            res__y = self.__process_wf(
                y=y,
                i_wf=i_wf,
                printPlot=printPlot,
                evalSat=evalSat, 
                log=log,
                prevSat=prevSat)
            
            # Unpack the results from the processed waveform
            peak_id__y, chisquare__y, all_pars__y, all_covs__y, max_curve__y, integ__y, dof__y, sat__y = res__y

            # Iterate through each peak identified in the waveform
            for j in range(len(peak_id__y)):
                results["blockID"].append(blockID)          # Assign blockID to each result
                results["wfID"].append(peak_id__y[j][0])    # Append waveform ID
                results["pkID"].append(peak_id__y[j][1])    # Append peak ID
                results["chi2"].append(chisquare__y[j])     # Append chi-squared value
                results["curveMax"].append(max_curve__y[j]) # Append curve maximum value
                results["integral"].append(integ__y[j])     # Append curve integral
                results["dof"].append(dof__y[j])            # Append degrees of freedom
                results["satCheck"].append(sat__y[j])       # Append saturation status
                
                # Append fit parameters with descriptive keys
                for k, v in zip(self.__par_names, all_pars__y[j]):
                    results[f'arg__{k}'].append(v) 
                
                # Append covariance matrix values for each parameter
                for k, v in zip(self.__par_names, all_covs__y[j]):
                    results[f'covs__{k}'].append(v) 
        
        ###########################
        #### CONVERT TO OUTPUT ####
        ###########################
        # Convert results dictionary to a pandas DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Add metadata as attributes to the DataFrame
        results_df.attrs['fName'] = self.fName
        results_df.attrs['detector'] = self.detector

        # Handle mode ('w' or 'a')
        if mode == 'w':
            self.results_df = results_df
        elif mode == 'a':
            if hasattr(self, 'results_df') and self.results_df is not None:
                self.results_df = pd.concat([self.results_df, results_df], ignore_index=True)
            else:
                self.results_df = results_df

        # Return the DataFrame containing all the results
        return self.results_df
#####################################################################################################################################################################
    def get_args_fitted(self):
        if self.results_df is None:
            raise Exception("Please fit a function by calling `process_arr`")
        columns_args = ['chi2', 'curveMax', 'integral', 'dof']
        # Trova tutte le colonne che contengono 'arg__' nel nome
        columns_args += [col for col in self.results_df.columns if 'arg__' in col]
        if not columns_args:
            raise Exception("No fitted arguments found in the results DataFrame.")
        # Restituisci solo le colonne corrispondenti
        return self.results_df[columns_args]
    
    def get_covs_fitted(self):
        if self.results_df is None:
            raise Exception("Please fit a function by calling `process_arr`")
        # Trova tutte le colonne che contengono 'arg__' nel nome
        columns_covs = [col for col in self.results_df.columns if 'covs__' in col]
        if not columns_covs:
            raise Exception("No fitted arguments found in the results DataFrame.")
        # Restituisci solo le colonne corrispondenti
        return self.results_df[columns_covs]
    
    def get_fitresults(self):
        if self.results_df is None:
            raise Exception("Please fit a function by calling `process_arr`")
        return self.results_df
    
#####################################################################################################################################################################
    def write_fitresults(self, file_path: str):
        """
        Save the fit results to an HDF5 file.

        Args:
            file_path (str): Path to the file where the fit results will be saved.

        Details:
        - The method saves the DataFrame `self.results_df` into an HDF5 file under the key `'dataframe'`.
        - It uses `h5py` to append metadata stored in `self.results_df.attrs` directly to the file's attributes.
        """
        print(f"Saving fit results in {file_path}...")
        # Save the DataFrame to an HDF5 file
        self.results_df.to_hdf(file_path, key='dataframe', mode='w')
        
        # Append metadata from the DataFrame attributes to the HDF5 file
        with h5py.File(file_path, 'a') as f:
            for key, value in self.results_df.attrs.items():
                f.attrs[key] = value

    def read_fitresults(self, file_path: str):
        """
        Load fit results and their metadata from an HDF5 file.

        Args:
            file_path (str): Path to the file from which the fit results will be read.

        Details:
        - The method loads the DataFrame stored under the key `'dataframe'`.
        - It also retrieves specific metadata (`'detector'` and `'fName'`) from the HDF5 file attributes 
        and assigns them to the DataFrame's attributes.
        """
        print(f"Reading fit results from {file_path}...")
        # Open the HDF5 file and load the DataFrame
        with h5py.File(file_path, 'r') as f:
            self.results_df = pd.read_hdf(file_path, key='dataframe')
            # Retrieve specific metadata and assign it to the DataFrame's attributes
            self.results_df.attrs['detector'] = f.attrs['detector']
            self.results_df.attrs['fName'] = f.attrs['fName']
            self.__parameters_checks(f.attrs['fName'], f.attrs['detector'])
        return self.results_df
    
    def to_h5(self, input_file: str, output_file: str,
              fName: Literal['double_exp', 'convolution_func', 'single_exp', 'orsa_function', 
                             'semigaussian_shaper', 'semigaussian_shaper_with_tail'] = 'double_exp', 
              detector: Literal['SiPM', 'PMT'] = 'SiPM'):
        """
        Converte un file di dati con una struttura specifica in un file CSV con colonne strutturate.

        Parameters:
            input_file (str): Percorso del file di input.
            output_file (str): Percorso del file CSV di output.
        """
        self.__parameters_checks(fName=fName, detector=detector)
        # Aprire il file e leggere le righe
        with open(input_file, 'r') as file:
            lines = file.readlines()
        # Lista per salvare i dati processati
        data = []
        # Analizzare le righe
        for line in lines:
            if line.startswith("###") or not line.strip():
                continue  # Ignorare commenti e righe vuote
            # Estrarre fullPeakID e i valori numerici
            split_line = line.split()
            full_peak_id = ast.literal_eval(split_line[0])  # Convertire in tupla
            values = list(map(float, split_line[1:]))  # Convertire i restanti valori in float
            # Separare wfID e pkID
            folder_id, file_id, wf_id, pk_id = full_peak_id
            # Aggiungere dati alla lista
            data.append([folder_id, file_id, wf_id, pk_id, *values])
        # Definire i nomi delle colonne
        columns = [
            'folder_id', 'file_id', 'wfID', 'pkID', 'chi2', 'curveMax', 'integral', 'dof',
        ] + [f'arg__{arg}' for arg in self.__par_names]
        # Creare un DataFrame pandas
        self.results_df = pd.DataFrame(data, columns=columns)
        # Aggiungere una colonna 'satCheck'
        self.results_df['satCheck'] = self.results_df['curveMax'] >= 8192 
        self.results_df.attrs['fName'] = fName
        self.results_df.attrs['detector'] = detector
        # Scrivere il DataFrame su file CSV
        self.write_fitresults(file_path=output_file)
        print(f"File HDF5 generato: {output_file}")

#####################################################################################################################################################################
    def make_plot(self, evalSat=False, quantile_thresh=0.95, log=False):
        fName = self.results_df.attrs['fName']
        detector = self.results_df.attrs['detector']
        # Ottieni i dati e i parametri
        df_args = self.get_args_fitted()  # DataFrame contenente i dati
        column_names = df_args.columns.to_numpy()  # Nomi delle colonne come array numpy
        val = df_args.to_numpy()  # Valori come array numpy (2D)
        # Rimuove la colonna 'satCheck' da par e val
        if 'satCheck' in column_names:
            # Trova l'indice della colonna 'satCheck' e rimuovila
            iSat = np.where(column_names == 'satCheck')[0][0]
            column_names = np.delete(column_names, iSat)
            val = np.delete(val, iSat, axis=1)
        spectra = ['curveMax', 'integral', 'A']
        #####################################
        #### dof<0 and inf chi2 removal  ####
        #####################################
        # Trova l'indice della colonna 'dof' e 'chi2'
        idof  = np.where(column_names == 'dof')[0][0]
        ichi2 = np.where(column_names == 'chi2')[0][0]   
        # Identifica le righe con dof < 0 o chi2 == inf
        nonpos_dof = np.where((val[:, idof] < 0) | np.isinf(val[:, ichi2]))[0]
        # Rimuovi queste righe da val
        clean_dof_val = np.delete(val, nonpos_dof, axis=0)
        if log:
            print(f'{len(nonpos_dof)} waveforms have been removed for dof < 0 or chi2 == inf.')
        #####################
        #### CHI2 CUTOFF ####
        #####################   
        # Calcola il cutoff del chi2
        chi2_thresh = np.quantile(clean_dof_val[:, ichi2], quantile_thresh)
        if log:
            print(f'Cutoff above chi2 values of {chi2_thresh}, which is the {quantile_thresh} quantile')
        # Identifica le righe con chi2 > chi2_thresh
        i_delete = np.where(clean_dof_val[:, ichi2] > chi2_thresh)[0]
        # Separare clean_val e unclean_val
        clean_val = np.delete(clean_dof_val, i_delete, axis=0)
        unclean_val = clean_dof_val[i_delete]
        # Crea i subplot
        fig, axs = plt.subplots(len(column_names), len(column_names), 
                                figsize=(2 * (len(column_names)), 
                                         2 * (len(column_names))))
        for par in range(len(column_names)):
            for j_par in range(len(column_names)):
                if j_par <= par:
                    if par == j_par:
                        # Istogrammi sulla diagonale
                        if column_names[par] in spectra:
                            hist_bins = np.logspace(
                                np.log10(max(min(clean_dof_val[:, par]), 1e-2)),
                                np.log10(max(clean_dof_val[:, par])),
                                600
                            )
                        else:
                            hist_bins = np.linspace(
                                min(clean_dof_val[:, par]),
                                max(clean_dof_val[:, par]),
                                600
                            )
                        axs[par][j_par].hist(clean_dof_val[:, par], bins=hist_bins, 
                                               log=True, color='deeppink')
                        axs[par][j_par].hist(clean_val[:, par], bins=hist_bins, 
                                               log=True, color='green')
                    else:
                        # Scatter plot sotto la diagonale
                        axs[par][j_par].scatter(unclean_val[:, j_par], unclean_val[:, par], 
                                                  s=2, color='deeppink')
                        axs[par][j_par].scatter(clean_val[:, j_par], clean_val[:, par], 
                                                  s=2, color='green')
                    # Etichette sugli assi
                    if par == len(column_names) - 1:
                        axs[par][j_par].set_xlabel(f'{column_names[j_par]}')
                    if j_par == 0:
                        axs[par][j_par].set_ylabel(f'{column_names[par]}')
                    axs[par][j_par].set_xticks([])
                    axs[par][j_par].set_yticks([])
                else:
                    axs[par][j_par].axis('off')

            if log:
                print(f'Line {[par]} done')
        fig.suptitle(f'Plots with threshold at chi2<{chi2_thresh:.2f} ({quantile_thresh} quantile)', y=0.9)
        # Mostra il plot
        if log:
            plt.show()

#####################################################################################################################################################################

    def __clean_fitresults(self, quantile_thresh=0.95, par_to_center=None, log=False):
        #####################################
        #### dof<0 and inf chi2 removal  ####
        #####################################
        # Make a copy of the results to avoid modifying the original DataFrame.
        clean_dof_val = self.results_df.copy()
        prev_len = len(clean_dof_val)

        # Remove rows where "dof" is infinite or less than or equal to zero.
        clean_dof_val = clean_dof_val[clean_dof_val["dof"] != np.inf]
        clean_dof_val = clean_dof_val[clean_dof_val["dof"] > 0]
        
        if log == True:
            # Log the number of removed rows due to invalid "dof" values.
            print(f'{prev_len - len(clean_dof_val)} waveforms have been removed for dof < 0 or chi2 == inf.')

        #####################
        #### CHI2 CUTOFF ####
        #####################

        if quantile_thresh > 1.:
            # If `quantile_thresh` is greater than 1, interpret it as an absolute value and 
            # compute the proportion of rows below this value.
            quantile_thresh = len(clean_dof_val[clean_dof_val['chi2'] < quantile_thresh]) / len(clean_dof_val)
        
        # Determine the chi2 cutoff value based on the quantile.
        chi2_thresh = np.quantile(clean_dof_val['chi2'], quantile_thresh)
        
        prev_len = len(clean_dof_val)   # Store the current length for logging.
        # Filter rows with chi2 values above the computed threshold.
        clean_val = clean_dof_val[clean_dof_val['chi2'] <= chi2_thresh]

        if log == True:
            # Log the number of removed rows and the chi2 threshold used.
            print(f'{prev_len - len(clean_val)} waveforms have been removed for Cutoff above chi2 values of {chi2_thresh}, which is the {quantile_thresh} quantile.')
        
        #############################
        #### PAR CENTERED CUTOFF ####
        #############################
        
        if par_to_center is not None:
            # Compute the mean and standard deviation for the parameter to center.
            center_average = np.average(clean_val[par_to_center])
            center_std = np.std(clean_val[par_to_center])

            prev_len = len(clean_val)   # Store the current length for logging.
            # Filter rows where the parameter is more than 3 standard deviations away from the mean.
            clean_val = clean_val[clean_val[par_to_center] >= center_average - 3 * center_std]
            clean_val = clean_val[clean_val[par_to_center] <= center_average + 3 * center_std]

            if log == True:
                # Log the number of removed rows due to the parameter cutoff.
                print(f'{prev_len - len(clean_val)} waveforms have been removed for |{par_to_center} - average({par_to_center})| > 3 std({par_to_center}).')
        
        return clean_dof_val, clean_val, chi2_thresh

    def params_analysis(self, quantile_thresh=0.95, par_to_center=None):
        """
        Analyze parameter data with filtering and visualization.

        This function processes a dataset by applying specific cutoff conditions, 
        optionally centers the analysis on a parameter, and generates histograms 
        for visualization. It can log details about each step for transparency.

        Parameters:
        ----------
        evalSat : bool
            Flag to indicate whether saturation checks should be considered during analysis.
        
        quantile_thresh : float, optional
            Threshold for chi2 cutoff, given as a quantile (default is 0.95). Values above this 
            quantile are removed. If greater than 1, it is interpreted as an absolute cutoff value.
        
        par_to_center : str, optional
            Name of the parameter to center the analysis on. The parameter's values are filtered 
            to within 3 standard deviations from the mean. Default is None.

        Returns:
        --------
        None
            The function modifies the internal dataset and visualizes the results 
            but does not return any value.
        """
        ################################
        #### TAKE RESULTS FROM FILE ####
        ################################
        # Log the initial details of the analysis if logging is enabled.
        if log == True:
            print(f'Detector: {self.detector}\nFunction {self.fName} with parameters: {self.par_names}')

        # If a parameter to center is specified, ensure it exists in the dataset.
        if par_to_center is not None:
            if par_to_center not in self.par_names:
                # Log an error if the parameter is invalid and exit the function.
                print(f'Parameter {par_to_center} not found.\nPossible parameters for function {self.fName} are {self.par_names}.')
                return

        ######################
        #### APPLY CUTOFF ####
        ######################
        clean_dof_val, clean_val, chi2_thresh = self.__clean_fitresults(quantile_thresh=quantile_thresh, par_to_center=par_to_center,  log=log)
        
        ####### PLOT FROM READ_ALLDATA_FIT #######
        # Parameters to apply logarithmic binning (for spectra-like parameters).
        spectra = ['curveMax', 'integral', 'arg__A']
        df_args = self.get_args_fitted()
        
        # Iterate over all columns in the DataFrame (excluding metadata columns).
        for par in df_args.columns:
            if par in ['folder_id', 'file_id', 'wfID', 'pkID', 'dof','satCheck']:
                continue    # Skip metadata columns.

            # Determine the histogram bins: logarithmic for spectral parameters, linear otherwise.
            if par in spectra:
                hist_bins = np.logspace(
                    np.log10(max(min(clean_dof_val[par]),1e2)),
                    np.log10(max(clean_dof_val[par])),
                    600
                )
                clean_bins = np.logspace(
                    np.log10(max(min(clean_val[par]),1e2)),
                    np.log10(max(clean_val[par])),
                    600
                )
            else:
                hist_bins = np.linspace(min(clean_dof_val[par]), max(clean_dof_val[par]), 600)
                clean_bins = np.linspace(min(clean_val[par]), max(clean_val[par]), 600)
            
            # Create a two-panel plot for the current parameter.
            fig, axs = plt.subplots(1,2, figsize=(16,4))
            # Histogram of all data (before cutoff) and filtered data (overlapping).
            axs[0].hist(clean_dof_val[par], bins=hist_bins, log=True, color = 'deeppink')
            axs[0].hist(clean_val[par], bins=hist_bins, log=True, color = 'green')
            # Histogram of filtered data only.
            axs[1].hist(clean_val[par], bins=clean_bins, log=True, color = 'green')
        
            # Add titles and formatting.
            plt.suptitle(f'{par}')
            axs[0].set_title('{} all data and below cutoff (chi2 < {:.2f})'.format(par, chi2_thresh))
            axs[1].set_title('{} below cutoff only (chi2 < {:.2f})'.format(par, chi2_thresh))
            
            if par in spectra:  # Apply log scale to the x-axis for spectral parameters.
                axs[0].set_xscale('log')
                axs[1].set_xscale('log')
            plt.show()

#####################################################################################################################################################################

    def get_spectrum_peak(self, curve_max, smoothing_radius = 5, der_fit_radius = 2, falsePositiveAcceptance = 0.05, log=False):
        # Generate logarithmic bins for the input data
        curve_bins = np.logspace(np.log10(min(curve_max)),
                                 np.log10(max(curve_max)),
                                 600
                     )
        curve_hist, _ = np.histogram(curve_max, bins=curve_bins)

        # Select the relevant range for the spectrum by focusing on the highest density region
        start_radius = 100
        start_center = 200
        start_spectrum = np.argmax(curve_hist[start_center - start_radius : start_center + start_radius]) + (start_center - start_radius)

        # Identify the end of the spectrum by finding the first zero after the peak
        try:
            first_zero = np.where(curve_hist[start_spectrum:] == 0)[0][0] + start_spectrum
        except:
            first_zero = len(curve_hist) - 1
            
        end_spectrum = min(len(curve_hist)-1,first_zero)
        
        # Reduce the histogram to the relevant range
        curve_hist_redux = curve_hist[start_spectrum:end_spectrum]
        
        # Smooth the spectrum and find the local extrema
        smooth_spectrum, max_points, min_points = self.find_extremes(curve_hist_redux, 
                                                                     smoothing_radius=smoothing_radius, 
                                                                     der_radius=der_fit_radius)

        # Adjust extrema positions based on smoothing parameters
        max_points = np.subtract(max_points, smoothing_radius+der_fit_radius)
        min_points = np.subtract(min_points, smoothing_radius+der_fit_radius)
        
        # Ensure equal numbers of max and min points by truncating the excess
        while not len(min_points) == len(max_points):
            if len(min_points) > len(max_points):
                min_points = np.delete(min_points, -1)
            else:
                max_points = np.delete(max_points, -1)
        
        # Define peak ends based on asymmetry assumptions 
        peak_asymmetry = 0.5
        peak_end = [min_points[i_peak]+int((2+peak_asymmetry)*(max_points[i_peak]-min_points[i_peak])) for i_peak in range(len(min_points))]

        # Create a flattened curve with peaks removed
        flat_curve = [x for x in curve_hist_redux]
        for k in range(len(curve_hist_redux)):
            for i_min in range(len(min_points)):
                if (k > min_points[i_min] + smoothing_radius + der_fit_radius) & (k <= peak_end[i_min] + smoothing_radius + der_fit_radius):
                    x_start = min_points[i_min] + der_fit_radius
                    y_start = smooth_spectrum[x_start]
                    
                    try:
                        x_end = peak_end[i_min] + der_fit_radius
                        y_end = smooth_spectrum[x_end]
                    except:
                        x_end = len(smooth_spectrum)-1
                        y_end = smooth_spectrum[x_end]
                    
                    # Linear interpolation to flatten the peak
                    flat_curve[k] = (y_end - y_start) / (x_end - x_start) * (k - x_start) + y_start

        # If log is True, visualize the spectrum and smoothing steps
        if log == True:
            fig, axs = plt.subplots(1,2, figsize=(16,4))
            axs[0].plot(range(len(curve_hist)), curve_hist)
            axs[0].axvline(start_spectrum, color='g', linestyle='--')
            axs[0].axvline(end_spectrum, color='r', linestyle='--')
            axs[0].set_yscale('log')
            axs[1].plot(range(len(curve_hist_redux)), curve_hist_redux)
            axs[1].set_yscale('log')
            plt.show()
            
            plt.figure()
            plt.plot(range(len(curve_hist_redux)), curve_hist_redux)
            plt.plot(np.add(range(len(smooth_spectrum)),smoothing_radius), smooth_spectrum)
            
            for i_max in range(len(max_points)):
                plt.axvspan(xmin=min_points[i_max]+der_fit_radius+smoothing_radius, 
                            xmax=peak_end[i_max]+der_fit_radius+smoothing_radius, 
                            color='g', alpha=0.2)
                plt.axvline(max_points[i_max]+der_fit_radius+smoothing_radius,color='r',linestyle='--')
            plt.yscale('log')
            plt.show()
        
        # Estimate background from the flattened curve
        bkg=[ this_bkg for this_bkg in flat_curve]
        smooth_bkg = [np.divide(np.sum(bkg[k-smoothing_radius:k+smoothing_radius]), 2 * smoothing_radius) 
                      for k in np.add(range(len(bkg) - 2 * smoothing_radius),smoothing_radius)]
        bkg_std = np.sqrt(np.divide(np.sum([(bkg[k + smoothing_radius] - smooth_bkg[k]) ** 2 
                                            for k in range(len(smooth_bkg))]), len(smooth_bkg)))
        
        # If log is True, visualize the background estimation
        if log == True:
            plt.figure()
            plt.plot(range(len(curve_hist_redux)), curve_hist_redux)
            plt.plot(range(len(smooth_bkg)), smooth_bkg)
            plt.show()

        # Subtract the background from the spectrum
        spectrum_bkgsub = np.subtract(curve_hist_redux[smoothing_radius:-smoothing_radius], smooth_bkg)
    
        # Initialize variables for peak fitting
        fitPeakPosition = []
        fitPeakSigma = []
        fitPeakA = []
        fitPeakMax = []
        fitPeakSignificance = []
        peak_fit_radius = 30    # Heuristic value for peak fitting radius
        areTherePeaks = True
        atPeak = 0

        # Iteratively fit and remove peaks
        while areTherePeaks == True:
            try:
                # Find the most prominent peak in the background-subtracted spectrum
                best_peak = np.argmax(spectrum_bkgsub)
                
                # Determine the left and right limits of the peak
                if best_peak-peak_fit_radius > 0:
                    left_limit = np.where(spectrum_bkgsub[best_peak-peak_fit_radius:best_peak] == 
                                          min(spectrum_bkgsub[best_peak-peak_fit_radius:best_peak]))[0][0] + best_peak-peak_fit_radius
                else:
                    left_limit = np.where(spectrum_bkgsub[:best_peak] == min(spectrum_bkgsub[:best_peak]))[0][0]
    
                if best_peak+peak_fit_radius < len(spectrum_bkgsub) - 1:
                    right_limit = np.where(spectrum_bkgsub[best_peak:best_peak+peak_fit_radius] == 
                                           min(spectrum_bkgsub[best_peak:best_peak+peak_fit_radius]))[0][0] + best_peak
                else:
                    right_limit = np.where(spectrum_bkgsub[best_peak:] == min(spectrum_bkgsub[best_peak:]))[0][0] + best_peak

                # Isolate the peak and fit a Gaussian
                isolate_peak = spectrum_bkgsub[left_limit:right_limit]
                starting_pars = [len(isolate_peak) / 2., len(isolate_peak) / 8., max(spectrum_bkgsub) * 20]
                [x0_peak, sigma_peak, A_peak], rest = curve_fit(lambda x, x0, std, A: A*stats.norm.pdf(x, loc=x0, scale=std), 
                                                                range(len(isolate_peak)), isolate_peak, p0=starting_pars)
                        
                best_peak_curve = [A_peak*stats.norm.pdf(x,loc=x0_peak,scale = sigma_peak) for x in range(len(isolate_peak))]

                # Update the peak information
                this_peak_position = x0_peak + left_limit
                this_peak_max = max(best_peak_curve)
                left_side = spectrum_bkgsub[:left_limit]
                middle = np.subtract(isolate_peak,best_peak_curve)
                right_side = spectrum_bkgsub[right_limit:]
                curve_peak_removed = np.concatenate((left_side,np.concatenate((middle,right_side)))) 

                # Calculate peak significance and validate
                thisPeakSignificance = (this_peak_max-curve_peak_removed[int(this_peak_position)])/bkg_std
                falsePeakProbability = stats.norm.sf(thisPeakSignificance)
    
                if falsePeakProbability < falsePositiveAcceptance and not A_peak < 0:
                    atPeak = atPeak + 1
                    fitPeakPosition.append(this_peak_position+smoothing_radius)
                    fitPeakSigma.append(sigma_peak)
                    fitPeakA.append(A_peak)
                    fitPeakMax.append(this_peak_max)
                    fitPeakSignificance.append(thisPeakSignificance)

                    # Optional logging and visualization
                    if log == True:
        
                        fig_thispeak, axs_thispeak = plt.subplots(1,2, figsize=(16,4))
                        axs_thispeak[0].plot(range(len(isolate_peak)), isolate_peak)
                        axs_thispeak[0].plot(range(len(isolate_peak)), 
                                             [starting_pars[2]*stats.norm.pdf(x, loc=starting_pars[0], scale=starting_pars[1]) 
                                              for x in range(len(isolate_peak))], color='g')
                        axs_thispeak[0].plot(range(len(best_peak_curve)), best_peak_curve, color='m')
                        axs_thispeak[0].set_title(f'Peak {atPeak} fit')
                        
                        axs_thispeak[1].plot(range(len(spectrum_bkgsub)), spectrum_bkgsub, color='deeppink')
                        axs_thispeak[1].plot(range(len(curve_peak_removed)), curve_peak_removed, color='green')
                        axs_thispeak[1].plot(range(len(curve_peak_removed)), [5*bkg_std for x in range(len(curve_peak_removed))], color='orange', linestyle = ':')
                        axs_thispeak[1].axvline(best_peak, color='r', linestyle = '--')
                        axs_thispeak[1].axvspan(xmin=left_limit, xmax=right_limit, color='g', alpha=0.2)
                        axs_thispeak[1].set_title(f'Spectrum with peaks up to {atPeak} subtracted')

                        plt.show()
        
                        print('Peak {} with center at {:.2f} and max = {:.2f} has been subtracted.'.format(atPeak,this_peak_position,this_peak_max))
                        print('Probability of false peak (white noise bkg assumption) {:.2f}'.format(falsePeakProbability))
                    
                    # Remove the fitted peak from the spectrum
                    spectrum_bkgsub = curve_peak_removed
                else:
                    # If the false positive probability exceeds the threshold, stop looking for peaks
                    areTherePeaks = False
            except:
                # Catch any exceptions and stop the peak fitting process
                areTherePeaks = False

        # Check if any peaks were successfully identified
        if len(fitPeakPosition) > 0 :
            if log == True:
                # Log the summary of the remaining peaks and their properties
                print(f'---\nRemaining peaks have false positive probability > {falsePositiveAcceptance}')
                print(f'The first peak FWHM is {2*np.sqrt(2*np.log(2))*fitPeakSigma[0]}')

                # Visualize the spectrum with the identified peaks
                plt.figure()
                plt.plot(range(len(curve_hist_redux)), curve_hist_redux)
                
                # Mark the identified peak positions
                for i_peak in range(len(fitPeakPosition)):
                    plt.axvline(fitPeakPosition[i_peak], color='r',linestyle='--')
                plt.yscale('log')
                plt.title('Identified peaks')
                plt.show()

                # Print the sorted peaks with their indices
                print('Sorted peaks')
                print('idx\t Peak position')
                for i_peak, peakPos in enumerate(np.sort(fitPeakPosition)):
                    print('{}\t{:.2f}'.format(i_peak,peakPos))
            
            # Return the identified peak properties
            return fitPeakPosition, fitPeakSigma, fitPeakA, fitPeakMax, fitPeakSignificance

        else:
            # If no peaks were found, log a message if logging is enabled
            if log == True:
                print(f'---\nNo peak could be fit. Returning position of spectrum maximum.')
            
            # Return the position of the spectrum's maximum as a fallback
            return best_peak
        

    def spectrum_analysis(self, smoothing_radius=5, quantile_thresh=0.95, par_to_center=None, 
                          der_fit_radius=2, falsePositiveAcceptance=0.05, fullOutput = False, 
                          log=False):
        ################################
        #### TAKE RESULTS FROM FILE ####
        ################################
        # Log the initial details of the analysis if logging is enabled.
        if log == True:
            print(f'Detector: {self.detector}\nFunction {self.fName} with parameters: {self.par_names}')

        # If a parameter to center is specified, ensure it exists in the dataset.
        if par_to_center is not None:
            if par_to_center not in self.par_names:
                # Log an error if the parameter is invalid and exit the function.
                print(f'Parameter {par_to_center} not found.\nPossible parameters for function {self.fName} are {self.par_names}.')
                return

        ######################
        #### APPLY CUTOFF ####
        ######################
        _, clean_val, _ = self.__clean_fitresults(quantile_thresh=quantile_thresh, par_to_center=par_to_center,  log=log)

        spectrum_peaks = self.get_spectrum_peak(clean_val['curveMax'], smoothing_radius, der_fit_radius, falsePositiveAcceptance, log)

        if len(spectrum_peaks) > 1:
            [peakPos, peakSigma, peakA, peakMax, peakSignificance] = spectrum_peaks
            
            if fullOutput == False:
                if log == True:
                    print(f'Returning FWHM of the first peak')
                return 2*np.sqrt(2*np.log(2))*peakSigma[0]
    
            else:
                if log == True:
                    print(f'Returning all fit and derived parameters for each peak: position, sigma, amplitude, maximum and significance')
                return peakPos, peakSigma, peakA, peakMax, peakSignificance
        else:
            spectrum_max = spectrum_peaks

            print(f'ALERT: NO PEAKS FOUND. Returning position of spectrum maximum.')
            
            return spectrum_max