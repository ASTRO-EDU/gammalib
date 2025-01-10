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



sys.path.append('/home/gamma/workspace/gammalib')
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
            [m_fit, q_fit], rest = curve_fit(lambda x, m, q: m*x+q, range(2*der_radius), smooth_array[fit_center - der_radius : fit_center + der_radius])
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
                            fName: Literal['double_exp', 'convolution_func', 'single_exp', 'orsa_function', 'semigaussian_shaper', 'semigaussian_shaper_with_tail'] = 'double_exp', 
                            detector: Literal['SiPM', 'PMT'] = 'SiPM'):
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
            self.par_names = [name for name in self.fToFit.__code__.co_varnames[2:]] #### parameters of input function 
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
                    dof_tmp = len(xToFit)-len(self.par_names) #len(arrToFit)-len(par_names)
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
            if log == True:
                print('No peaks found.')
        return peak_id, chisquare, all_pars, all_covs, max_curve, integ, dof, satCheck

#####################################################################################################################################################################
    ####
    #### takes one file and returns fit parameters for all peaks in it
    def process_arr(self, 
                    data: npt.NDArray[np.int_],
                    fName: Literal['double_exp', 'convolution_func', 'single_exp', 'orsa_function', 'semigaussian_shaper', 'semigaussian_shaper_with_tail'] = 'double_exp', 
                    detector: Literal['SiPM', 'PMT'] = 'SiPM', 
                    evalSat: bool = False, 
                    printPlot: bool = False,
                    startEvent: int = 0, 
                    endEvent: int = -1,
                    log: bool = False):
        
        # INPUT PARAMETER CHECKS 
        self.__parameters_checks(fName, detector=detector)
        #######################
        #### RETURN VALUES ####
        #######################
        results = []
        if log == True:
            print(f'##############################\nFitting function {self.fToFit.__name__} on a {detector} entire tensor')
        #####
        results = {
            "wfID": [],
            "pkID": [],
            "chi2": [],
            "curveMax": [],
            "integral": [],
            "dof": [],
            "satCheck": [],
        }
        # Add fit parameters with descriptive keys
        for k in self.par_names:
            results[f'arg__{k}'] = []
        # Add covariance matrix of parameters
        for k in self.par_names:
            results[f'covs__{k}'] = []
        #####
        data_section = data[startEvent:endEvent]
        prevSat = False
        for y, i in zip(data_section, range(len(data_section))):
            i_wf = i + startEvent
            # Process the waveform
            res__y = self.__process_wf(
                y=y,
                i_wf=i_wf,
                printPlot=printPlot,
                evalSat=evalSat, 
                log=log,
                prevSat=prevSat)
            peak_id__y, chisquare__y, all_pars__y, all_covs__y, max_curve__y, integ__y, dof__y, sat__y = res__y
            for j in range(len(peak_id__y)):
                results["wfID"].append(peak_id__y[j][0]),
                results["pkID"].append(peak_id__y[j][1]),
                results["chi2"].append(chisquare__y[j]),
                results["curveMax"].append(max_curve__y[j]),
                results["integral"].append(integ__y[j]),
                results["dof"].append(dof__y[j]),
                results["satCheck"].append(sat__y[j]),
                # Add fit parameters with descriptive keys
                for k, v in zip(self.par_names, all_pars__y[j]):
                    results[f'arg__{k}'].append(v) 
                # Add covariance matrix of parameters
                for k, v in zip(self.par_names, all_covs__y[j]):
                    results[f'covs__{k}'].append(v) 
        # Convert the list of dictionaries to a pandas DataFrame
        results_df = pd.DataFrame(results)
        results_df.attrs['fName'] = self.fName
        results_df.attrs['detector'] = self.detector
        self.results_df = results_df
        return results_df
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
        ] + [f'arg__{arg}' for arg in self.par_names]
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
        par = df_args.columns.to_numpy()  # Nomi delle colonne come array numpy
        val = df_args.to_numpy()  # Valori come array numpy (2D)
        # Rimuove la colonna 'satCheck' da par e val
        if 'satCheck' in par:
            # Trova l'indice della colonna 'satCheck' e rimuovila
            iSat = np.where(par == 'satCheck')[0][0]
            par = np.delete(par, iSat)
            val = np.delete(val, iSat, axis=1)
        spectra = ['curveMax', 'integral', 'A']
        #####################################
        #### dof<0 and inf chi2 removal  ####
        #####################################
        # Trova l'indice della colonna 'dof' e 'chi2'
        idof  = np.where(par == 'dof')[0][0]
        ichi2 = np.where(par == 'chi2')[0][0]   
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
        fig, axs = plt.subplots(len(par), len(par), 
                                figsize=(2 * (len(par)), 
                                         2 * (len(par))))
        for i_par in range(len(par)):
            for j_par in range(len(par)):
                if j_par <= i_par:
                    if i_par == j_par:
                        # Istogrammi sulla diagonale
                        if par[i_par] in spectra:
                            hist_bins = np.logspace(
                                np.log10(max(min(clean_dof_val[:, i_par]), 1e-2)),
                                np.log10(max(clean_dof_val[:, i_par])),
                                600
                            )
                        else:
                            hist_bins = np.linspace(
                                min(clean_dof_val[:, i_par]),
                                max(clean_dof_val[:, i_par]),
                                600
                            )
                        axs[i_par][j_par].hist(clean_dof_val[:, i_par], bins=hist_bins, 
                                               log=True, color='deeppink')
                        axs[i_par][j_par].hist(clean_val[:, i_par], bins=hist_bins, 
                                               log=True, color='green')
                    else:
                        # Scatter plot sotto la diagonale
                        axs[i_par][j_par].scatter(unclean_val[:, j_par], unclean_val[:, i_par], 
                                                  s=2, color='deeppink')
                        axs[i_par][j_par].scatter(clean_val[:, j_par], clean_val[:, i_par], 
                                                  s=2, color='green')
                    # Etichette sugli assi
                    if i_par == len(par) - 1:
                        axs[i_par][j_par].set_xlabel(f'{par[j_par]}')
                    if j_par == 0:
                        axs[i_par][j_par].set_ylabel(f'{par[i_par]}')
                    axs[i_par][j_par].set_xticks([])
                    axs[i_par][j_par].set_yticks([])
                else:
                    axs[i_par][j_par].axis('off')

            if log:
                print(f'Line {[i_par]} done')
        fig.suptitle(f'Plots with threshold at chi2<{chi2_thresh:.2f} ({quantile_thresh} quantile)', y=0.9)
        # Mostra il plot
        if log:
            plt.show()