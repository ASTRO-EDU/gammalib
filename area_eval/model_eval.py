from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from math import ceil 
from scipy.stats import norm
from scipy.optimize import curve_fit
import pandas as pd
import numpy.typing as npt
from typing import Union

class ModelEval:
    def __init__(self):
        # Initialize alg
        self.alg    = None
        self.res_dict = {}
        self.colors      = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        self.linescolors = ['red', 'blue', 'orange', 'blue']
        self.linestyles  = ['solid', 'dotted', 'dashed', 'dashdot']
        # Definiamo una mappa di colori per ogni picco
        self.colormaps   = [cm.Blues, cm.Oranges, cm.Greens, cm.Reds, cm.Purples]
        # Base colormap colors
        self.base_colormaps = [cm.Blues, cm.Oranges, cm.Greens, cm.Reds, cm.Purples]


    def add_results(self, y_real: npt.ArrayLike, *y_preds: npt.ArrayLike, methods: Union[str, list[str]]):
        if isinstance(methods, str):
            methods = [methods]

        if len(y_preds) != len(methods):
            raise ValueError("Il numero di y_pred deve essere uguale al numero di methods")

        self.y_real    = y_real
        self.y_preds   = {alg: y_pred for alg, y_pred in zip(methods, y_preds)}
        self.alg_names = list(self.y_preds.keys())
        # Pad y_real and y_pred to have same shapes
        self.__prepare_labels()
        # Prepare the result dict for results table
        if isinstance(methods, list):
            self.res_dict.update({
                alg: {
                    f"peak_{idx_peak + 1}": {} for idx_peak in range(self.y_real.shape[-1])
                } for alg in methods
            })
        else:
            self.res_dict.update({
                methods: {
                    f"peak_{idx_peak + 1}": {} for idx_peak in range(self.y_real.shape[-1])
                }
            })
        print(self.res_dict)
        

    ##############################################################################################################################

    def __generate_colormaps(self, num_shades: int) -> List:
        """
        Generates distinct shades of colors using predefined colormaps.

        Args:
            num_shades (int): Number of shades per colormap.

        Returns:
            List of colors extracted from the colormaps.
        """
        colormap_list = []
        for cmap in self.base_colormaps:
            colormap_list.extend([cmap(i) for i in np.linspace(0.3, 0.9, num_shades)])  # Pick colors from the middle shades
        return colormap_list
    

    def __prepare_labels(self):
        self.max_len = max(self.y_real.shape[-1], *(y_pred.shape[-1] for y_pred in self.y_preds.values()))
        # Padding y_real
        self.y_real = np.pad(
            self.y_real, ((0, 0), (0, self.max_len - self.y_real.shape[-1])),
            mode='constant', constant_values=1e-9
        )
        # Padding di tutti i y_pred nel dizionario
        self.y_preds = {
            method: np.pad(
                y_pred, ((0, 0), (0, self.max_len - y_pred.shape[-1])),
                mode='constant', constant_values=1e-9
            )
            for method, y_pred in self.y_preds.items()
        }


    def __diff_sign(self, y_real, y_pred):
        return y_real - y_pred

    def __filter_goodarea(self, y_real, y_pred): 
        return y_real[
            np.logical_and(
                y_pred[:,0] > 1,
                y_pred[:,1] > 1)
            ]

    def __difference_relative_ratio(self, y_real, y_pred):
        """
        Area Relative Ratio:
        """
        # Padding di self.y_real e self.y_pred per farli avere la stessa shape
        return self.__diff_sign(y_real, y_pred) / y_real

    def __mean_diff_rel_ratio(self, y_real, y_pred):
        """
        Mean Area Relative Ratio:
        """
        return np.mean(np.abs(self.__diff_sign(y_pred, y_real) / y_real))

    def __npreds_over_nreal(self, y_real, y_pred):
        """
        Number predictions over Number real:
        """
        # Filtra self.y_pred per includere solo gli elementi che hanno la stessa lunghezza degli elementi corrispondenti in self.y_real
        filtered_y_pred = [ap for ap, ar in zip(y_pred, y_real) if len(ap) == len(ar)]
        # Calcola il rapporto tra la lunghezza di filtered_y_pred e la lunghezza di self.y_real
        return len(filtered_y_pred) / len(y_real)

    def __get_alg(self, idx_alg: int):
        """
        Retrieves the algorithm instance if it is already set; otherwise, returns the algorithm name
        corresponding to the provided index.

        Args:
            idx_alg (int): Index of the algorithm in the `alg_names` list.

        Returns:
            object or str: The algorithm instance (`self.alg`) if it is not None; otherwise, 
            the algorithm name from `self.alg_names[idx_alg]`.
        """
        if self.alg is None:
            return self.alg_names[idx_alg]
        else:
            return self.alg

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################


    def eval(self, path=None, lim=1000,   
             alg='method_to_eval',
             plot_list=["diff_rel_ratio_hists",
                        "diff_rel_ratio_boxplot",
                        "diff_rel_ratio_perc",
                        "hists",
                        "relative_difference_histogram",
                        "absolute_difference_2d_histogram"],
            size=(10, 6), ncol=2, wspace=0.3, hspace=0.3):
        # Initialize figure parameter
        self.size   = size
        self.ncol   = ncol
        self.wspace = wspace
        self.hspace = hspace
        
        self.plot_list = plot_list
        self.total_plots = len(plot_list)
        self.nrows = int(np.ceil((self.max_len * self.total_plots) / self.ncol))
        
        # Initialize figure
        self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncol, figsize=self.size)
        self.fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        self.axes = self.axes.flatten()

        # Extract y_pred for current alg
        self.alg = alg
        self.y_pred = self.y_preds[self.alg]
        
        # Initialize result dictionary for current alg
        self.res_dict[self.alg].update({
            "MRR": self.__mean_diff_rel_ratio(self.y_real, self.y_pred), # Mean Relative Ratio
            "MAE": np.mean(np.abs(self.y_real - self.y_pred)),  # Mean Absolute Error
            "MSE": np.mean((self.y_real - self.y_pred) ** 2)    # Mean Squared Error
        })
        
        print(f'Mean Relative Ratio {self.res_dict[self.alg]["MRR"]}')
        
        self.fig.suptitle(f"Model Evaluation - {self.alg}")
        self.index_subplot = 0

        self.plot__difference_relative_ratio(bin_size=0.01, ylogscale=True)
        self.plot__diff_rel_ratio_boxplot()
        self.plot__diff_rel_ratio_perc(bin_size=0.001, ylogscale=True)
        self.plot__hists(logscale=True, title=f'diff_rel_cutscale_(logscale) {self.alg}')
        self.plot_relative_difference_histogram(bin_size1D=0.01, logscale=True)
        self.plot_absolute_difference_2d_histogram()

        plt.show()

        return None
    


    def compare(self, size=(10, 6),  
                ncol=2, wspace=0.3, hspace=0.3,
                plot_list=["diff_rel_ratio_hists",
                           "diff_rel_ratio_boxplot",
                           "diff_rel_ratio_perc",]):
        self.alg = None

        self.size   = size
        self.ncol   = ncol
        self.wspace = wspace
        self.hspace = hspace
        self.total_plots = len(plot_list)
        
        self.plot_list = plot_list
        # Calcola la somma dei valori corrispondenti alle chiavi in plot_list
        self.nrows = int(np.ceil((self.max_len * self.total_plots) / self.ncol))

        self.fig, self.axes = plt.subplots(nrows=self.nrows, ncols=self.ncol, figsize=self.size)
        self.fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        self.axes = self.axes.flatten()
        self.fig.suptitle(f"Model Evaluation - Comparison")
        self.index_subplot = 0
        
        self.y_pred = [self.y_preds[method] for method in self.y_preds.keys()]

        self.plot__difference_relative_ratio(bin_size=0.2, ylogscale=True)
        self.plot__diff_rel_ratio_boxplot()
        self.plot__diff_rel_ratio_perc(bin_size=0.2, ylogscale=True)
        # self.plot__hists(logscale=True)
        

    def get_comptable(self):
        """
        Crea una tabella di confronto da un dizionario di risultati.
        
        Returns:
            pd.DataFrame: Tabella di confronto organizzata per metrica, picco e modello.
        """
        # Lista per raccogliere i dati in formato tabellare
        rows = []

        # Itera attraverso i modelli nel dizionario
        for model_name, model_data in self.res_dict.items():
            # Aggiungi la metrica globale 'MRR'
            # Aggiungi i dati relativi a ciascun picco
            for peak_key, peak_data in model_data.items():
                if peak_key.startswith("peak_"):
                    for metric, value in peak_data.items():
                        rows.append({
                            "Model": model_name,
                            "Metric": metric,
                            "Peak": peak_key,
                            "Value": value
                        })
                else:
                    metric_name = peak_key
                    value       = peak_data
                    rows.append({
                        "Model": model_name,
                        "Metric": metric_name,
                        "Peak": "Global",
                        "Value": value
                    })

        # Crea il DataFrame da tutte le righe raccolte
        df = pd.DataFrame(rows)

        # Riorganizza i dati per una visualizzazione più intuitiva
        df_pivot = df.pivot(index=["Metric", "Peak"], columns="Model", values="Value")
        df_pivot = df_pivot.sort_index(level=["Metric", "Peak"])
        df_pivot = df_pivot.style.apply(
            lambda row: [
                "font-weight: bold" if (isinstance(v, (int, float))) and (abs(v) == np.abs(row).min()) else ""
                for v in row
            ],
            axis=1
        )
        return df_pivot
    
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################


    def plot__difference_relative_ratio(self, bin_size=0.1, xlogscale=False, ylogscale=True):
        """
        Plots histograms and overlays results if multiple y_pred are provided.
        Different peaks use different color gradients.
        """
        # Se y_pred è un singolo array, lo trasformiamo in lista
        if not isinstance(self.y_pred, list):
            F_whisckers = True
            y_preds_list = [self.y_pred]
        else:
            F_whisckers = False
            y_preds_list = self.y_pred

        global_min = float(np.inf)
        global_max = float(-np.inf)

        # Determiniamo il minimo e massimo globale tra tutte le y_pred
        all_diff_values = []
        for y_pred in y_preds_list:
            diff_values = self.__difference_relative_ratio(y_real=self.y_real, y_pred=y_pred)
            all_diff_values.append(diff_values)
            global_min = min(global_min, np.min(diff_values))
            global_max = max(global_max, np.max(diff_values))

        # Creazione dei bin comuni a tutti gli istogrammi
        bins = np.linspace(global_min, global_max + bin_size, num=100)
        # bins = 100

        for idx_peak in range(self.y_real.shape[-1]):
            ax_hist = self.axes[self.index_subplot]
            self.index_subplot += 1

            cmap = self.colormaps[idx_peak % len(self.colormaps)]  # Se ci sono più picchi dei colori predefiniti, si riutilizzano

            for idx_pred, diff_values in enumerate(all_diff_values):
                hist, bin_edges = np.histogram(diff_values[:, idx_peak], bins=bins, density=True)
                hist = hist / np.sum(hist)

                color = cmap(0.3 + 0.7 * idx_pred / (len(y_preds_list) - 1)) if len(y_preds_list) > 1 else cmap(0.6)
                ls    = self.linestyles[idx_pred]
                hatch = ['/', '\\', '-'][idx_pred]
                
                # cm = self.__generate_colormaps(3)
                # color = self.colors[idx_pred]

                ax_hist.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), label=f'diff {idx_peak+1}° peak ({self.__get_alg(idx_pred)})',
                            alpha=0.5, ecolor=color, edgecolor='w')
                # ax_hist.plot(bin_edges[:-1], hist, color=color, linestyle=ls, alpha=0.6, label=f'diff {idx_peak+1}° peak ({self.__get_alg(idx_pred)})')

                # Calcolo dei quartili per whiskers e median
                p25 = np.percentile(diff_values[:, idx_peak], 25)
                p50 = np.percentile(diff_values[:, idx_peak], 50)
                p75 = np.percentile(diff_values[:, idx_peak], 75)
                iqr = p75 - p25
                min_whisker = p25 - 1.5 * iqr
                max_whisker = p75 + 1.5 * iqr

                if F_whisckers:
                    ax_hist.axvline(min_whisker, color=color, linestyle='--', linewidth=1, label=f'whiskers {idx_pred+1}')
                    ax_hist.axvline(max_whisker, color=color, linestyle='--', linewidth=1)
                    ax_hist.axvline(p50, color=color, linestyle='-', linewidth=1, label=f'median {idx_pred+1}')

            ax_hist.set_title(f'Histogram diff_rel_ratio {idx_peak+1}° peak')
            if ylogscale:
                ax_hist.set_yscale('log')
            if xlogscale:
                ax_hist.set_xscale('symlog')
            ax_hist.set_ylabel('normalized counts')
            ax_hist.set_xlabel('Relative Difference Ratio: (y_real - y_pred) / y_real')
            ax_hist.legend()
        

    def plot__diff_rel_ratio_boxplot(self):
        """
        Plots boxplots of diff_rel_ratio and prints the Mean diff_rel_ratio in the title.
        Supports multiple y_pred overlays in the same subplot.
        """
        if not isinstance(self.y_pred, list):
            y_preds_list = [self.y_pred]
        else:
            y_preds_list = self.y_pred
        
        all_diff_values = []
        for y_pred in y_preds_list:
            diff_values = self.__difference_relative_ratio(y_real=self.y_real, y_pred=y_pred)
            all_diff_values.append(diff_values)
        
        for idx_peak in range(self.y_real.shape[-1]):
            ax_boxplt = self.axes[self.index_subplot]
            self.index_subplot += 1
            
            cmap = self.colormaps[idx_peak % len(self.colormaps)]
            box_colors = [cmap(0.3 + 0.7 * idx / (len(y_preds_list) - 1)) if len(y_preds_list) > 1 else cmap(0.6)
                          for idx in range(len(y_preds_list))]
            
            box_data = [diff_values[:, idx_peak] for diff_values in all_diff_values]
            
            bp = ax_boxplt.boxplot(box_data, patch_artist=True)
            for patch, _ in zip(bp['boxes'], box_colors):
                patch.set_facecolor('white')
            
            ax_boxplt.set_title(f'Boxplot diff_rel_ratio {idx_peak+1}° peak')
            ax_boxplt.set_ylabel('Relative Difference Ratio')
            ax_boxplt.set_xticklabels([f'{self.__get_alg(idx)}' for idx in range(len(y_preds_list))])


    

    def plot__diff_rel_ratio_perc(self, bin_size=0.1, xlogscale=False, ylogscale=True):
        """ 
        Plots histograms of diff_rel_ratio with key percentiles, supporting multiple y_pred overlays.
        """
        # Se y_pred è un singolo array, lo trasformiamo in lista
        if not isinstance(self.y_pred, list):
            F_singleAlg = True
            y_preds_list = [self.y_pred]
        else:
            F_singleAlg = False
            y_preds_list = self.y_pred

        global_min = float(np.inf)
        global_max = float(-np.inf)

        # Determiniamo il minimo e massimo globale tra tutte le y_pred
        all_diff_values = []
        for y_pred in y_preds_list:
            diff_values = self.__difference_relative_ratio(y_real=self.y_real, y_pred=y_pred)
            diff_values = np.clip(diff_values, 0, 1)
            all_diff_values.append(diff_values)
            global_min = min(global_min, np.min(diff_values))
            global_max = max(global_max, np.max(diff_values))

        # Creazione dei bin comuni a tutti gli istogrammi
        bins = np.linspace(global_min, global_max + bin_size, num=100)
        
        for idx_peak in range(self.y_real.shape[-1]):
            ax = self.axes[self.index_subplot]
            self.index_subplot += 1
            cmap = self.colormaps[idx_peak % len(self.colormaps)]  # Colori per il picco
            
            for idx_pred, diff_values in enumerate(all_diff_values):
                p_list = [5, 25, 50, 75, 95] if F_singleAlg else [95]
                percentiles = np.percentile(diff_values[:, idx_peak], p_list)

                bins = np.linspace(diff_values[:, idx_peak].min(), diff_values[:, idx_peak].max() + bin_size, num=100)
                hist, bin_edges = np.histogram(diff_values[:, idx_peak], bins=bins, density=True)
                hist /= hist.sum()
                
                color = cmap(0.3 + 0.7 * idx_pred / (len(y_preds_list) - 1)) if len(y_preds_list) > 1 else cmap(0.6)
                ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), 
                       # color="white", edgecolor=color, 
                       alpha=0.5, ecolor=color, edgecolor='w',
                       label=f'diff {idx_peak+1}° peak ({self.__get_alg(idx_pred)})')
                    
                for perc, c, label in zip(percentiles, [self.linescolors[idx_pred]] * len(p_list), [f'perc_{p:02d} ({self.__get_alg(idx_pred)})' for p in p_list]):
                    ax.axvline(perc, color=c, linestyle='--', linewidth=1, label=f"{label}: {perc:.5f}")
                    self.res_dict[self.__get_alg(idx_pred)][f"peak_{idx_peak + 1}"].update({
                        f'absdrr__{label.split()[0]}': round(perc, 5)
                        })
            if ylogscale:
                ax.set_yscale('log')
            if xlogscale:
                ax.set_xscale('symlog')
            
            ax.set_ylabel('Normalized Counts')
            ax.set_xlabel('Relative abs Difference (clip[0, 1]): |y_real - y_pred| / y_real')
            ax.legend()
            ax.set_title(f'diff_rel_ratio_clip[0,1] + percentiles on {idx_peak+1}° peak')
        
        return None

    def plot__hists(self, bin_size=0.01, logscale=False, title='Diff_relative_cutscl'):
        """
        Adds histograms of the relative difference between real and predicted values to self.axes.
        Supports multiple y_pred overlays in the same subplot.
        Computes and stores statistical metrics in self.res.

        Args:
            bin_size (float, optional): Bin width for histogram. If None, it is computed automatically.
            logscale (bool, optional): If True, uses log scale for y-axis.
            title (str): Title of the plot.
        """
        if not isinstance(self.y_pred, list):
            y_preds_list = [self.y_pred]
        else:
            y_preds_list = self.y_pred
        
        colormaps = [cm.Blues, cm.Oranges, cm.Greens, cm.Reds, cm.Purples]  # Diverse tonalità per ogni picco
        all_diff_values = []
        
        for y_pred in y_preds_list:
            diff_values = self.__difference_relative_ratio(y_real=self.y_real, y_pred=y_pred)
            diff_values = np.clip(diff_values, -1, 1)  # Limit outliers
            all_diff_values.append(diff_values)
        
        for idx_peak in range(self.y_real.shape[-1]):
            ax = self.axes[self.index_subplot]
            self.index_subplot += 1
            cmap = colormaps[idx_peak % len(colormaps)]
            
            for idx_pred, diff_values in enumerate(all_diff_values):
                hist, bin_edges = np.histogram(diff_values[:, idx_peak], bins=int(2//bin_size), range=(-1, 1), density=True)
                hist /= hist.sum()
                
                color = cmap(0.3 + 0.7 * idx_pred / (len(y_preds_list) - 1)) if len(y_preds_list) > 1 else cmap(0.6)
                ax.bar(bin_edges[:-1], hist, width=bin_size, color=color, alpha=0.5, 
                       label=f'diff {idx_peak+1}° peak ({self.__get_alg(None)})')
                
                # Compute statistics
                diff_mean = diff_values[:, idx_peak].mean()
                diff_median = np.median(diff_values[:, idx_peak])
                diff_std = diff_values[:, idx_peak].std()
                
                self.res_dict[self.__get_alg(idx_pred)][f"peak_{idx_peak + 1}"].update({
                    f"drr__mean": round(diff_mean, 5),
                    f"drr__median": round(diff_median, 5),
                    f"drr__std": round(diff_std, 5)
                })
                
            if logscale:
                ax.set_yscale('log')
            
            ax.set_xlabel('Relative Difference (clip[-1, 1]): (y_real - y_pred) / y_real')
            ax.set_ylabel('Normalized Counts')
            ax.set_title(f'diff_rel_ratio_clip[-1,1] on {idx_peak+1}° peak')
            ax.legend()


    def plot_relative_difference_histogram(self, bin_size1D=0.01,
                                           logscale=False):
        """
        Plot the relative difference histogram with Gaussian fit for a specific peak.
        """
        def gaussian_fit(x, mu, sigma):
            return norm.pdf(x, mu, sigma)
        
        abs_diff = self.__diff_sign(self.y_real, self.y_pred)
        diff_relative = abs_diff / self.y_real
        diff_relative = np.clip(diff_relative, -1, 1)  # Limit outliers

        for idx_peak in range(diff_relative.shape[-1]):
            ax1 = self.axes[self.index_subplot]
            self.index_subplot += 1

            hist, bin_edges = np.histogram(diff_relative[:, idx_peak], bins=int(2//bin_size1D), range=(-1, 1), density=True)
            initial_guess = [np.mean(diff_relative), np.std(diff_relative)]
            params, _ = curve_fit(gaussian_fit, bin_edges[:-1], hist, p0=initial_guess)
            mu_fit, sigma_fit = params
            bell_curve = gaussian_fit(bin_edges[:-1], mu_fit, sigma_fit)

            epsilon = 1e-10
            chi_squared = np.sum(((bell_curve - hist) ** 2) / (hist + epsilon))
            
            self.res_dict[self.__get_alg(None)][f"peak_{idx_peak + 1}"].update({
                "fit_mu": mu_fit,
                "fit_sigma": sigma_fit,
                "fit_chi_squared": chi_squared
            })
            
            hist /= np.sum(hist)
            bell_curve /= np.sum(bell_curve)
            
            ax1.bar(bin_edges[:-1], hist, width=bin_size1D, alpha=0.5, color=self.colors[idx_peak], label='Data')
            ax1.plot(bin_edges[:-1], bell_curve, color='red', label='Gaussian Fit')
            ax1.plot([], [], color='white', label=f'Mean: {mu_fit:.5f}\nSigma: {sigma_fit:.5f}\nChi^2: {chi_squared:.5f}')
            
            if logscale:
                ax1.set_yscale('log')
                min_value = np.min(hist[hist > 0])
                ax1.set_ylim([10 ** np.floor(np.log10(min_value)), 1.1])
            
            ax1.set_xlabel('Relative Difference: (y_real - y_pred) / y_real')
            ax1.set_ylabel('Normalized Counts')
            ax1.set_title(f'Relative Difference Histogram & Gaussian Fit - Peak {idx_peak+1}')
            ax1.legend()

    def plot_absolute_difference_2d_histogram(self, bin_size2D_x=None, bin_size2D_y=None, logscale=False, 
                                              hist2D_colors=['Blues', 'Oranges', 'Greens']):
        """
        Plot the 2D histogram of absolute differences for a specific peak.
        """
        abs_diff = self.__diff_sign(self.y_real, self.y_pred)

        for idx_peak in range(abs_diff.shape[-1]):
            x_min, x_max = np.min(abs_diff[:, idx_peak]), abs_diff[:, idx_peak].max()
            y_min, y_max = np.min(self.y_real), np.max(self.y_real)
            
            if bin_size2D_x is None:
                bin_size2D_x = (x_max - x_min) / 100
            if bin_size2D_y is None:
                bin_size2D_y = (y_max - y_min) / 100
            
            x_bins = np.arange(x_min, x_max + bin_size2D_x, bin_size2D_x)
            y_bins = np.arange(y_min, y_max + bin_size2D_y, bin_size2D_y)
            
            counts, _, _ = np.histogram2d(abs_diff[:, idx_peak].flatten(), self.y_real[:, idx_peak].flatten(), bins=[x_bins, y_bins])
            
            if logscale:
                counts = np.log1p(counts)
            
            ax2 = self.axes[self.index_subplot]
            self.index_subplot += 1
            
            img = ax2.imshow(counts.T, origin='lower', cmap=hist2D_colors[idx_peak],
                            extent=[x_min, x_max, y_min, y_max], aspect='auto')
            plt.colorbar(img, ax=ax2, label='Counts (log scale)' if logscale else 'Counts')
            
            ax2.set_xlabel('Absolute difference (y_real - y_pred)')
            ax2.set_ylabel('Real Values (y_real)')
            ax2.set_title(f'2D Histogram of Absolute Differences - Peak {idx_peak+1}')



    ##############################################################################################################################

    def plot__example(self, data, index):
        ax = self.axes[index % self.ncol]  # Se index eccede, si riutilizzano gli assi ciclicamente
        ax.plot(data, label="Example Plot")
        ax.legend()
        ax.set_title(f"Plot {index+1}")
        return ax
    
    def show_plots(self):
        plt.tight_layout()
        plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    model_eval = ModelEval(size=(12, 6), method="MyMethod", ncol=3)
    data = np.random.rand(100)
    for i in range(3):
        model_eval.plot__example(data * (i+1), i)
    model_eval.show_plots()