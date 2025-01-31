from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil 
from scipy.stats import norm
from scipy.optimize import curve_fit
import pandas as pd



##############################################################################################################################

def diff_sign(y_real, y_pred):
    return y_real - y_pred

##############################################################################################################################

def filter_goodarea(y_real, y_pred): 
  return y_real[np.logical_and(
                          y_pred[:,0] > 1,
                          y_pred[:,1] > 1)
                  ]

##############################################################################################################################

def difference_relative_ratio(y_real, y_pred):
    """
    Area Relative Ratio:
    """
    # Padding di y_real e y_pred per farli avere la stessa shape
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), mode='constant', constant_values=1e-9)
    return diff_sign(y_real, y_pred) / (y_real+1e-9)




def mean_difference_relative_ratio(y_real, y_pred):
    """
    Mean Area Relative Ratio:
    """
    return np.mean(np.abs(diff_sign(y_pred, y_real) / (y_real + 1e-9)))

##############################################################################################################################

def npreds_over_nreal(y_real, y_pred):
    """
    Number predictions over Number real:
    """
    # Filtra y_pred per includere solo gli elementi che hanno la stessa lunghezza degli elementi corrispondenti in y_real
    filtered_y_pred = [ap for ap, ar in zip(y_pred, y_real) if len(ap) == len(ar)]
    # Calcola il rapporto tra la lunghezza di filtered_y_pred e la lunghezza di y_real
    return len(filtered_y_pred) / len(y_real)




def plot__difference_relative_ratio(y_real, y_pred, bin_size=0.1, path=None, xlogscale=False, ylogscale=True, 
                                    showfliers=True, title='Histogram difference_relative_ratio'):
    """
    Plots histograms and boxplot difference_relative_ratio and print the Mean difference_relative_ratio in the title
    """
    # Padding di y_real e y_pred per farli avere la stessa shape
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), 
                        mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), 
                        mode='constant', constant_values=1e-9)

    for idx_peak in range(y_real.shape[-1]):
        # Calcolo di Mean_difference_relative_ratio e difference_relative_ratio
        mean_difference_relative_ratio_value = mean_difference_relative_ratio(y_real=y_real[:, idx_peak], 
                                                                              y_pred=y_pred[:, idx_peak])
        difference_relative_ratio_values = difference_relative_ratio(y_real=y_real[:, idx_peak], 
                                                                     y_pred=y_pred[:, idx_peak])

        # Definisci il numero di bin e il range
        # steps = int((difference_relative_ratio_values.max() - difference_relative_ratio_values.min()) // bin_size)
        # bins = np.linspace(-1, 1+bin_size, steps)
        bins = np.linspace(difference_relative_ratio_values.min(),
                           difference_relative_ratio_values.max()+bin_size, 
                           num=100)
        # bins = min(
        #         ceil(
        #             (difference_relative_ratio_values.max() - difference_relative_ratio_values.min()) / bin_size
        #        ), 100)

        # Creazione dei subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Calcola gli istogrammi
        hist1, bin_edges = np.histogram(difference_relative_ratio_values, 
                                        bins=bins, density=True)
        hist1 = hist1 / np.sum(hist1)
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][idx_peak]  # Seleziona il colore

        # Plot degli istogrammi utilizzando plt.bar()
        axs[0].bar(bin_edges[:-1], hist1, width=np.diff(bin_edges), color=color, label=f'diff {idx_peak+1}° peak')
        # axs[0].hist(hist1, bins=bin_edges[:-1], color=color, label=f'diff {idx_peak+1}° peak')
        axs[0].set_title(f'Histogram difference_relative_ratio {idx_peak+1}° peak')
        # axs[0].set_xlim((-1-bin_size, 1+bin_size))
        if ylogscale:
            axs[0].set_yscale('log')
        if xlogscale:
            axs[0].set_xscale('symlog')
        axs[0].set_ylabel('normalized counts')
        axs[0].set_xlabel('Relative Difference Ratio: (y_real - y_pred) / y_real')

        # Plot del boxplot
        res = axs[1].boxplot(difference_relative_ratio_values, showfliers=showfliers)
        axs[1].set_title(f'Boxplot difference_relative_ratio {idx_peak+1}° peak')
        axs[1].set_yscale('symlog')

        # Aggiungi linee verticali per i baffi
        median = np.median(difference_relative_ratio_values)
        whiskers = res["whiskers"]  # I whiskers contengono i valori minimo e massimo
        min_whisker = whiskers[0].get_ydata()[1]  # Estremo inferiore
        max_whisker = whiskers[1].get_ydata()[1]  # Estremo superiore

        axs[0].axvline(min_whisker, color='red', linestyle='--', linewidth=1, label="whiskers")
        axs[0].axvline(max_whisker, color='red', linestyle='--', linewidth=1)
        axs[0].axvline(median, color='orange', linestyle='--', linewidth=1, label='median')
        axs[0].legend()

        # Aggiunta di Mean_difference_relative_ratio come testo nel subplot dell'istogramma
        fig.suptitle(f'{title} on {idx_peak+1}° peak: {mean_difference_relative_ratio_value:.5f}', fontsize=14)

        # Visualizzazione del plot
        plt.tight_layout()
        if path is not None:
            # Crea ricorsivamente il percorso se non esiste
            os.makedirs(path, exist_ok=True)
            # Salva la figura nella cartella desiderata
            plt.savefig(os.path.join(path, f'difference_relative_ratio_plot__{idx_peak+1}.png'))
        plt.show()



def plot_difference_relative_ratio_with_percentiles(y_real, y_pred, bin_size=0.1, path=None, 
                                                    xlogscale=False, ylogscale=True, 
                                                    title='Histogram difference_relative_ratio'):
    """
    Plots histograms of difference_relative_ratio and adds vertical lines for key percentiles.
    Removes the boxplot.

    Args:
        y_real (np.ndarray): Real values.
        y_pred (np.ndarray): Predicted values.
        bin_size (float, optional): Bin width for histogram.
        path (str, optional): Path to save the plot.
        xlogscale (bool): If True, uses symmetric log scale for x-axis.
        ylogscale (bool): If True, uses log scale for y-axis.
        title (str): Title of the plot.
    """
    # Padding di y_real e y_pred per avere la stessa shape
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), 
                        mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), 
                        mode='constant', constant_values=1e-9)

    for idx_peak in range(y_real.shape[-1]):
        difference_relative_ratio_values = np.abs(
            difference_relative_ratio(y_real=y_real[:, idx_peak], 
                                      y_pred=y_pred[:, idx_peak])
        )
        difference_relative_ratio_values[difference_relative_ratio_values > 1] = 1

        # Calcolo percentili
        percentiles = np.percentile(difference_relative_ratio_values, [5, 25, 50, 75, 95])
        p5, p25, p50, p75, p95 = percentiles

        # Definisci il numero di bin e il range
        bins = np.linspace(difference_relative_ratio_values.min(),
                           difference_relative_ratio_values.max() + bin_size, 
                           num=100)

        # Creazione del subplot
        fig, ax = plt.subplots(figsize=(6, 4))

        # Calcola l'istogramma
        hist, bin_edges = np.histogram(difference_relative_ratio_values, bins=bins, density=True)
        hist /= np.sum(hist)  # Normalizzazione

        # Seleziona il colore
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][idx_peak]

        # Plot dell'istogramma
        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color=color, alpha=0.6, label=f'diff {idx_peak+1}° peak')

        # Aggiunta delle linee verticali ai percentili
        ax.axvline(p5, color='red', linestyle='--', linewidth=1, label="5° / 95° percentile")
        ax.axvline(p95, color='red', linestyle='--', linewidth=1)

        ax.axvline(p25, color='blue', linestyle='--', linewidth=1, label="25° / 75° percentile")
        ax.axvline(p75, color='blue', linestyle='--', linewidth=1)

        ax.axvline(p50, color='orange', linestyle='--', linewidth=1, label='50° percentile (median)')

        # Opzioni di scala
        if ylogscale:
            ax.set_yscale('log')
        if xlogscale:
            ax.set_xscale('symlog')

        ax.set_ylabel('Normalized Counts')
        ax.set_xlabel('Relative Difference: (y_real - y_pred) / y_real')
        # ax.set_title(f'Histogram difference_relative_ratio {idx_peak+1}° peak')
        ax.legend()

        # Aggiunta di Mean_difference_relative_ratio come testo
        fig.suptitle(f'{title} on {idx_peak+1}° peak', fontsize=14)

        # Salvataggio della figura
        if path is not None:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f'difference_relative_ratio_plot_{idx_peak+1}.png'))

        # Mostra il plot
        plt.show()




def false_prediction_rate(y_real, y_pred, alpha: float = 0.5):
    """
    False Area Prediction Relative Rate:
    """
    # Padding di y_real e y_pred per farli avere la stessa shape
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), mode='constant', constant_values=1e-9)
    difference_relative_ratio = np.abs(diff_sign(y_real, y_pred) / (y_real + 1e-9))
    fpred_len = len(difference_relative_ratio[difference_relative_ratio > alpha])
    return fpred_len/len(difference_relative_ratio)




def print__false_prediction_rate(y_real, y_pred, alpha_list: List, res: Optional[dict] = None) -> dict:
    """
    Calcola e stampa i tassi di predizione errata per ogni picco e per ciascun valore di alpha.
    Inoltre, raccoglie i risultati in un dizionario, che può essere passato come argomento.

    Args:
        y_real (np.ndarray): Array bidimensionale dei valori reali (forma: [n_samples, n_peaks]).
        y_pred (np.ndarray): Array bidimensionale delle predizioni (forma: [n_samples, n_peaks]).
        alpha_list (List[float]): Lista dei valori di alpha da considerare per il calcolo.
        res (Optional[dict]): Dizionario opzionale in cui aggiungere i risultati. 
                              Se non fornito, viene creato uno nuovo.

    Returns:
        dict: Dizionario contenente i tassi di predizione errata per ogni picco e ogni valore di alpha.
              La struttura è: 
              {
                  "peak_1": {"fpr_alpha_0.1": valore, "fpr_alpha_0.2": valore, ...},
                  "peak_2": {"fpr_alpha_0.1": valore, "fpr_alpha_0.2": valore, ...},
                  ...
              }
    """
    # Determina la lunghezza massima tra y_real e y_pred
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    # Ensure the results dictionary is initialized
    if res is None:
        res = {}
    for idx_peak in range(max_len):
        res[f"peak_{idx_peak + 1}"] = {}

    # Esegue il padding di y_real se necessario
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), mode='constant', constant_values=1e-9)

    # Esegue il padding di y_pred se necessario
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), mode='constant', constant_values=1e-9)

    # Itera su ogni picco
    for idx_peak in range(y_real.shape[-1]):
        peak_key = f"peak_{idx_peak + 1}"  # Nome del picco
        res[peak_key] = res.get(peak_key, {})  # Inizializza il dizionario per il picco se non esiste

        # Calcola e registra i tassi di predizione errata per ogni alpha
        for alpha in alpha_list:
            rate = false_prediction_rate(y_real=y_real[:, idx_peak], y_pred=y_pred[:, idx_peak], alpha=alpha)
            res[peak_key][f"fpr_alpha_{alpha}"] = rate  # Salva il risultato nel dizionario
            print(f'False Prediction Rate Integral on {idx_peak + 1}° peak (alpha={alpha}): {rate:.5f}')
        print()

    return res




def difference_overmean_error(y_real, y_pred):
    """
    Area Difference Over Mean Real Area percentage error
    """
    diff = np.abs(diff_sign(y_real, y_pred))
    return np.mean(diff)/np.mean(y_real)*100




def print__difference_over_mean(y_real, y_pred, res: Optional[dict] = None) -> dict:
    """
    Calcola e stampa l'errore percentuale dell'Area Difference Over Mean Real Area 
    per ogni picco. I risultati sono inoltre salvati in un dizionario.

    Args:
        y_real (np.ndarray): Array bidimensionale dei valori reali (forma: [n_samples, n_peaks]).
        y_pred (np.ndarray): Array bidimensionale delle predizioni (forma: [n_samples, n_peaks]).
        res (Optional[dict]): Dizionario opzionale in cui aggiungere i risultati. 
                              Se non fornito, viene creato uno nuovo.

    Returns:
        dict: Dizionario contenente gli errori percentuali dell'Area Difference Over Mean Real Area.
              La struttura è:
              {
                  "peak_1": {"dfovmean": valore},
                  "peak_2": {"dfovmean": valore},
                  ...
              }
    """
    # Determina la lunghezza massima tra y_real e y_pred
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    
    # Ensure the results dictionary is initialized
    if res is None:
        res = {}
    for idx_peak in range(max_len):
        res[f"peak_{idx_peak + 1}"] = {}

    # Esegue il padding di y_real se necessario
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), mode='constant', constant_values=1e-9)

    # Esegue il padding di y_pred se necessario
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), mode='constant', constant_values=1e-9)

    # Itera su ogni picco
    for idx_peak in range(y_real.shape[-1]):
        # Calcola l'errore percentuale per il picco corrente
        error = difference_overmean_error(y_real=y_real[:, idx_peak], y_pred=y_pred[:, idx_peak])
        
        # Salva il risultato nel dizionario
        res[f"peak_{idx_peak + 1}"]["dfovmean"] = error
        
        # Stampa il risultato
        print(f'Area over mean on {idx_peak + 1}° peak: {error:.5f}')

    return res


def plot__hists(y_real, y_pred,
                title='Diff_relative_cutscl',
                new_max=None, new_min=None,
                bin_size=None, logscale=False,
                path=None, res: Optional[Dict] = None) -> Dict:
    """
    Crea e stampa istogrammi della differenza relativa normalizzata tra valori reali e predetti.
    I risultati statistici (media, mediana, deviazione standard) vengono salvati in un dizionario.

    Args:
        y_real (np.ndarray): Array bidimensionale dei valori reali (forma: [n_samples, n_peaks]).
        y_pred (np.ndarray): Array bidimensionale delle predizioni (forma: [n_samples, n_peaks]).
        title (str): Titolo degli istogrammi.
        new_max (float, optional): Valore massimo del range degli istogrammi. Default calcolato.
        new_min (float, optional): Valore minimo del range degli istogrammi. Default calcolato.
        bin_size (float, optional): Dimensione dei bin per gli istogrammi. Default calcolato.
        logscale (bool, optional): Se True, usa una scala logaritmica sull'asse y. Default False.
        path (str, optional): Percorso per salvare i grafici. Default None (non salva).
        res (Optional[Dict]): Dizionario opzionale per salvare i risultati. 
                              Se non fornito, ne viene creato uno nuovo.

    Returns:
        Dict: Dizionario contenente statistiche calcolate per ogni picco. La struttura è:
              {
                  "peak_1": {"drr__mean": valore, "drr__median": valore, "drr__std": valore},
                  "peak_2": {"drr__mean": valore, "drr__median": valore, "drr__std": valore},
                  ...
              }
    """
    # Determina la lunghezza massima tra y_real e y_pred
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    
    # Ensure the results dictionary is initialized
    if res is None:
        res = {}
    for idx_peak in range(max_len):
        res[f"peak_{idx_peak + 1}"] = {}

    # Padding degli array se necessario
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), mode='constant', constant_values=1e-9)

    # Calcola la differenza relativa normalizzata
    diff = difference_relative_ratio(y_real, y_pred)
    diff[diff > 1.] = 1.  # Limita gli outliers superiori
    diff[diff < -1.] = -1.  # Limita gli outliers inferiori

    # Calcola automaticamente new_max e new_min se non specificati
    if new_max is None:
        new_max = diff[diff < 1].max() * 1.618  # Esclude gli outliers, usa il golden ratio
    if new_min is None:
        new_min = diff[diff > -1].min() * 1.618  # Esclude gli outliers, usa il golden ratio

    # Controlla che il range sia valido
    if new_max <= new_min:
        raise ValueError(f"Invalid range: new_min={new_min}, new_max={new_max}")

    # Calcola automaticamente bin_size se non specificato
    if bin_size is None:
        bin_size = abs(new_max - new_min) / 200

    # Controlla che bin_size sia valido
    if bin_size <= 0 or np.isclose(bin_size, 0):
        raise ValueError(f"Invalid bin_size computed: {bin_size}. Check new_min={new_min}, new_max={new_max}")

    # Calcola il numero di bin
    num_bins = int((new_max - new_min) / bin_size)

    # Controlla che il numero di bin sia valido
    if num_bins <= 0:
        raise ValueError(f"Invalid num_bins computed: {num_bins}. Check new_min={new_min}, new_max={new_max}, bin_size={bin_size}")

    # Stampa titolo
    print(f'### {title}')

    # Calcola statistiche globali
    diff_mean = diff.mean(axis=0)
    diff_median = np.median(diff, axis=0)
    diff_std = diff.std(axis=0)
    print(f'{title} mean: {diff_mean}, std: {diff_std}')

    # Itera su ogni picco
    for idx_peak in range(diff.shape[-1]):
        # Calcola l'istogramma per il picco corrente
        hist1, bin_edges = np.histogram(diff[:, idx_peak], bins=num_bins, range=(new_min, new_max), density=True)
        hist1 = hist1 / np.sum(hist1)

        # Salva le statistiche nel dizionario
        res[f"peak_{idx_peak + 1}"]["drr__mean"]   = diff_mean[idx_peak]
        res[f"peak_{idx_peak + 1}"]["drr__median"] = diff_median[idx_peak]
        res[f"peak_{idx_peak + 1}"]["drr__std"]    = diff_std[idx_peak]

        # Plot dell'istogramma
        plt.bar(bin_edges[:-1], hist1, width=bin_size, alpha=0.5, label=f'diff {idx_peak + 1}° peak')
        if logscale:
            plt.yscale('log')

        # Etichetta statistica
        plt.plot([], [], color='white', 
                 label=f'μ_{idx_peak + 1}: {diff_mean[idx_peak]:.5f},\n'
                       f'σ_{idx_peak + 1}: {diff_std[idx_peak]:.5f}')

    # Configura il grafico
    plt.legend()
    plt.xlabel('Relative Difference Ratio: (y_real - y_pred) / y_real')
    plt.ylabel('Counts normalized')
    plt.title(f'{title} tra real e reco (bin_size={bin_size})')

    # Salva il grafico se specificato un percorso
    if path is not None:
        os.makedirs(path, exist_ok=True)  # Crea il percorso se non esiste
        plt.savefig(os.path.join(path, f'diff_relative.png'))

    # Mostra il grafico
    plt.show()

    return res




def plot__gaussian_fitted(y_real, y_pred, 
                          title='Diff_relative_cutscl vs Gaussian fit', 
                          new_max=None, new_min=None, bin_size=None, 
                          logscale=False, path=None, res=None):
    """
    Plot histograms of the relative difference between real and predicted values,
    fit a Gaussian distribution to the data, and compute statistics like mean, 
    standard deviation, and chi-squared.

    Parameters:
        y_real (np.ndarray): Array of real values.
        y_pred (np.ndarray): Array of predicted values.
        title (str): Title for the plots and results. Default is 'Diff_relative_cutscl vs Gaussian fit'.
        new_max (float, optional): Maximum range for the histogram. Default is calculated to zoom into the data.
        new_min (float, optional): Minimum range for the histogram. Default is calculated to zoom into the data.
        bin_size (float, optional): Bin size for the histogram. Default is determined based on the range.
        logscale (bool): Whether to use logarithmic scale for the y-axis. Default is False.
        path (str, optional): Directory to save the plots. If None, plots are not saved. Default is None.
        res (dict, optional): Dictionary to append results to. If None, a new dictionary is created.

    Returns:
        dict: A dictionary containing the computed results for each peak.
              Format: {peak_index: {"mu": ..., "sigma": ..., "chi_squared": ...}}
    """
    # Ensure y_real and y_pred have the same shape by padding with small constant values
    max_len = max(y_real.shape[-1], y_pred.shape[-1])
    
    # Ensure the results dictionary is initialized
    if res is None:
        res = {}
    for idx_peak in range(max_len):
        res[f"peak_{idx_peak + 1}"] = {}
        
    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), mode='constant', constant_values=1e-9)

    # Define Gaussian fitting function
    def gaussian_fit(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    # Compute relative difference between real and predicted values
    diff = diff_sign(y_real, y_pred) / (y_real + 1e-9)

    # Set range limits for the histogram
    if new_max is None:
        new_max = diff[diff < 1].max() * 1.618  # Use golden ratio for zoom
    if new_min is None:
        new_min = diff[diff > -1].min() * 1.618  # Use golden ratio for zoom

    # Compute bin size if not provided
    if bin_size is None:
        bin_size = abs(new_max - new_min) / 200

    # Group outliers into the extreme bins
    diff[diff > 1.] = 1.
    diff[diff < -1.] = -1.

    # Define number of bins and range
    num_bins = int((new_max - new_min) / bin_size)
    range_min, range_max = new_min, new_max

    # Iterate over each peak to compute and plot
    for idx_peak in range(diff.shape[-1]):
        # Extract data for the current peak
        data = diff[:, idx_peak]

        # Perform Gaussian fitting
        initial_guess = [np.mean(data), np.std(data)]
        hist, bin_edges = np.histogram(data, bins=num_bins, range=(range_min, range_max), density=True)
        params, covariance = curve_fit(gaussian_fit, bin_edges[:-1], hist, p0=initial_guess)
        mu_fit, sigma_fit = params

        # Compute Gaussian curve and normalize both hist and curve
        bell_curve = gaussian_fit(bin_edges[:-1], mu_fit, sigma_fit)
        hist /= np.sum(hist)
        bell_curve /= np.sum(bell_curve)

        # Calculate chi-squared
        epsilon = 1e-10
        chi_squared = np.sum(((bell_curve - hist) ** 2) / (hist + epsilon))

        # Save results in the dictionary
        res[f"peak_{idx_peak + 1}"]["fit_mu"]          = mu_fit
        res[f"peak_{idx_peak + 1}"]["fit_sigma"]       = sigma_fit
        res[f"peak_{idx_peak + 1}"]["fit_chi_squared"] = chi_squared

        # Plot histogram and Gaussian fit
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][idx_peak % 5]
        plt.bar(bin_edges[:-1], hist, width=bin_size, alpha=0.5, label='Data', color=color)
        plt.plot(bin_edges[:-1], bell_curve, color='red', label='Gaussian Fit')

        # Log scale adjustment if enabled
        if logscale:
            plt.yscale('log')
            min_value = np.min(hist[hist > 0])
            plt.ylim([10 ** np.floor(np.log10(min_value)), 1.1])

        # Annotate plot with results
        plt.plot([], [], color='white', label=f'Mean: {mu_fit:.5f}\nSigma: {sigma_fit:.5f}\nChi^2: {chi_squared:.5f}')
        plt.legend()
        plt.xlabel('Relative Difference Ratio: (y_real - y_pred) / y_real')
        plt.ylabel('Counts normalized')
        plt.title(f'{title} - Peak {idx_peak + 1}')

        # Save plot if path is provided
        if path:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f'diff_relative_Gfit__{idx_peak + 1}.png'))

        # Show the plot
        plt.show()

    return res


def plot__2d_bars(y_real, y_pred, 
                  title='Area Relative Ratio 2DHistogram', 
                  relative=False,
                  bin_size_x=None, bin_size_y=None, 
                  x_min=-1000, x_max=1000, 
                  cmap=['viridis', 'inferno'], logscale=False, 
                  path=None):
    """
    Crea un barplot bidimensionale per rappresentare i rapporti relativi delle aree, i valori effettivi
    e i conteggi in una mappa di colori.

    Args:
        y_real (np.nddifference_relative_ratioay): Aree reali (difference_relative_ratioay 1D).
        y_pred (np.nddifference_relative_ratioay): Aree predette (difference_relative_ratioay 1D, stessa lunghezza di y_real).
        title (str): Titolo del grafico.
        bin_size_x (float): Dimensione del bin per i rapporti relativi (ascisse).
        bin_size_y (float): Dimensione del bin per le aree reali (ordinate).
        cmap (str): Colormap per i conteggi.
        logscale (bool): Se True, applica la scala logaritmica ai conteggi.
        path (str): Cartella dove salvare il grafico (se specificata).

    Returns:
        None
    """
    # Assicuriamoci che gli input siano numpy array
    # y_real = np.asarray(y_real).flatten()
    # y_pred = np.asarray(y_pred).flatten()
    # Calcola i rapporti relativi
    relative_ratios = diff_sign(y_real, y_pred)
    if relative:
        relative_ratios = np.divide(relative_ratios, y_real + 1e-9, 
                                    out=np.zeros_like(relative_ratios), 
                                    where=(y_real != 0))
    # Calcola il range per i rapporti e le aree reali
    x_min, x_max = np.min(relative_ratios), relative_ratios.max()  # Limiti fissi per il rapporto
    # x_min, x_max = -1000, 1000  # Limiti fissi per il rapporto
    y_min, y_max = np.min(y_real), np.max(y_real)
    
    # Determina i bin
    if bin_size_x is None:
        bin_size_x = (x_max - x_min) / 100  # Default: 100 bins
    if bin_size_y is None:
        bin_size_y = (y_max - y_min) / 100  # Default: 100 bins
    # Calcolo dei bin edges
    x_bins = np.arange(x_min, x_max + bin_size_x, bin_size_x)
    y_bins = np.arange(y_min, y_max + bin_size_y, bin_size_y)
    for idx_peak in range(relative_ratios.shape[-1]):
        
        
        # Costruzione dell'istogramma bidimensionale
        counts, _, _ = np.histogram2d(relative_ratios[:, idx_peak].flatten(), 
                                      y_real[:, idx_peak].flatten(), 
                                      bins=[x_bins, y_bins])
        
        # Imposta il colore logaritmico, se richiesto
        if logscale:
            counts = np.log1p(counts)  # log(1 + counts) per evitare log(0)
        
        # Configurazione del plot
        plt.figure(figsize=(10, 8))
        plt.imshow(counts.T, origin='lower', cmap=cmap[idx_peak],
                   extent=[x_min, x_max, y_min, y_max],
                   aspect='auto')
        plt.colorbar(label='Counts (log scale)' if logscale else 'Counts')
        plt.xlabel('Relative Difference Ratio: (y_real - y_pred) / y_real' if relative else 'Difference: y_real - y_pred')
        plt.ylabel('Real Area')
        plt.title(f'{title} on {idx_peak+1}° peak')
        
        # Salvataggio del grafico
        if path is not None:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f"{title.replace(' ', '_')}__{idx_peak+1}.png"))
        
        # Mostra il grafico
        plt.show()


def plot__2d_bars_with_quartiles(y_real, y_pred, 
                                title='2D Histogram with Quartiles', 
                                relative=False, 
                                bin_size_x=None, bin_size_y=None, 
                                cmap='viridis', logscale=False, 
                                path=None):
    """
    Crea un barplot bidimensionale e calcola i quartili delle differenze assolute tra y_real e y_pred.

    Args:
        y_real (np.ndarray): Valori reali (array 1D).
        y_pred (np.ndarray): Valori predetti (array 1D, stessa lunghezza di y_real).
        title (str): Titolo del grafico.
        relative (bool): Se True, calcola il rapporto relativo (differenza/y_real).
        bin_size_x (float): Dimensione del bin per le ascisse.
        bin_size_y (float): Dimensione del bin per le ordinate.
        cmap (str): Colormap per i conteggi.
        logscale (bool): Se True, applica la scala logaritmica ai conteggi.
        path (str): Cartella dove salvare il grafico (se specificata).

    Returns:
        None
    """
    # Calcolo delle differenze assolute
    abs_differences = np.abs(y_real - y_pred)
    
    # Calcolo dei quartili
    q1, q2, q3 = np.percentile(abs_differences, [25, 50, 75])
    print(f"Quartili delle differenze assolute: Q1={q1}, Q2={q2} (mediana), Q3={q3}")
    
    # Calcolo del range per i dati
    x_min, x_max = np.min(y_pred), np.max(y_pred)
    y_min, y_max = np.min(y_real), np.max(y_real)
    
    # Determina i bin
    if bin_size_x is None:
        bin_size_x = (x_max - x_min) / 100  # Default: 100 bins
    if bin_size_y is None:
        bin_size_y = (y_max - y_min) / 100  # Default: 100 bins
    
    # Calcolo dei bin edges
    x_bins = np.arange(x_min, x_max + bin_size_x, bin_size_x)
    y_bins = np.arange(y_min, y_max + bin_size_y, bin_size_y)
    
    # Costruzione dell'istogramma bidimensionale
    counts, _, _ = np.histogram2d(y_pred, y_real, bins=[x_bins, y_bins])
    
    # Imposta il colore logaritmico, se richiesto
    if logscale:
        counts = np.log1p(counts)  # log(1 + counts) per evitare log(0)
    
    # Configurazione del plot
    plt.figure(figsize=(10, 8))
    plt.imshow(counts.T, origin='lower', cmap=cmap,
                extent=[x_min, x_max, y_min, y_max],
                aspect='auto')
    plt.colorbar(label='Counts (log scale)' if logscale else 'Counts')
    plt.xlabel('Predicted Values (y_pred)')
    plt.ylabel('Real Values (y_real)')
    plt.title(title)
    
    # Annotazione dei quartili
    for q, label in zip([q1, q2, q3], ['Q1', 'Q2 (Median)', 'Q3']):
        plt.axhline(q, color='red', linestyle='--', linewidth=1, label=f'{label}={q:.2f}')
    plt.legend()
    
    # Salvataggio del grafico
    if path is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{title.replace(' ', '_')}.png"))
    
    # Mostra il grafico
    plt.show()


def plot_combined_histograms(y_real, y_pred, 
                             title='Comparison of Relative and Absolute Differences', 
                             new_max=None, new_min=None, 
                             logscale=False, path=None, res=None,
                             bin_size1D=None, bin_size2D_x=None, bin_size2D_y=None, 
                             hist1D_colors=['tab:blue', 'tab:orange', 'tab:green'],
                             hist2D_colors=['Blues', 'Oranges', 'Greens']):
    """
    Genera un'unica figura con due subplot affiancati:
    - A sinistra: istogramma delle differenze relative con fit gaussiano (2/3 dello spazio)
    - A destra: istogramma 2D delle differenze assolute (1/3 dello spazio)

    Args:
        y_real (np.ndarray): Valori reali.
        y_pred (np.ndarray): Valori predetti.
        title (str): Titolo del grafico.
        new_max (float, optional): Valore massimo per il range dell'istogramma.
        new_min (float, optional): Valore minimo per il range dell'istogramma.
        bin_size (float, optional): Dimensione dei bin per l'istogramma delle differenze relative.
        logscale (bool): Se True, utilizza scala logaritmica per entrambi i subplot.
        path (str, optional): Cartella in cui salvare il grafico.
        res (dict, optional): Dictionary to append results to. If None, a new dictionary is created.
        bin_size1D (float): Dimensione del bin per i rapporti relativi.
        bin_size2D_x (float): Dimensione del bin per i rapporti assoluti (ascisse).
        bin_size2D_y (float): Dimensione del bin per le aree assoluti (ordinate).: 
        hist1D_colors (str): Colors per i conteggi dell'istogramma 1D.
        hist2D_colors (str): Colormap per i conteggi dell'istogramma 2D.

    Returns:
        dict: A dictionary containing the computed results for each peak.
              Format: {peak_index: {"mu": ..., "sigma": ..., "chi_squared": ...}}
    """    
    # Ensure y_real and y_pred have the same shape by padding with small constant values
    max_len = max(y_real.shape[-1], y_pred.shape[-1])

    # Ensure the results dictionary is initialized
    if res is None:
        res = {}
    for idx_peak in range(max_len):
        res[f"peak_{idx_peak + 1}"] = {}

    if y_real.shape[-1] < max_len:
        y_real = np.pad(y_real, ((0, 0), (0, max_len - y_real.shape[-1])), 
                        mode='constant', constant_values=1e-9)
    if y_pred.shape[-1] < max_len:
        y_pred = np.pad(y_pred, ((0, 0), (0, max_len - y_pred.shape[-1])), 
                        mode='constant', constant_values=1e-9)

    
    # Compute relative difference between real and predicted values
    abs_diff      = diff_sign(y_real, y_pred)
    diff_relative = diff_sign(y_real, y_pred) / y_real
    # Group outliers into the extreme bins
    diff_relative[diff_relative > 1.] = 1.
    diff_relative[diff_relative < -1.] = -1.

    # Fit gaussiano
    def gaussian_fit(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    # Iterate over each peak to compute and plot
    for idx_peak in range(abs_diff.shape[-1]):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), 
                                 gridspec_kw={'width_ratios': [4, 3]})
        
        ### --- SUBPLOT 1: ISTOGRAMMA DIFFERENZE RELATIVE + FIT GAUSSIANO --- ###
        ax1 = axes[0]
        
        # Determina il range dell'istogramma
        if new_max is None:
            new_max = diff_relative[diff_relative[:, idx_peak] < 1][:, idx_peak].max() * 1.618
        if new_min is None:
            new_min = diff_relative[diff_relative[:, idx_peak] > -1][:, idx_peak].min() * 1.618
        if bin_size1D is None:
            bin_size1D = abs(new_max - new_min) / 200

        num_bins = int((new_max - new_min) / bin_size1D)
        hist, bin_edges = np.histogram(diff_relative[:, idx_peak], 
                                       bins=num_bins, range=(new_min, new_max), density=True)

        initial_guess = [np.mean(diff_relative), np.std(diff_relative)]
        params, _ = curve_fit(gaussian_fit, bin_edges[:-1], hist, p0=initial_guess)
        mu_fit, sigma_fit = params
        bell_curve = gaussian_fit(bin_edges[:-1], mu_fit, sigma_fit)

        # Calculate chi-squared
        epsilon = 1e-10
        chi_squared = np.sum(((bell_curve - hist) ** 2) / (hist + epsilon))

        # Save results in the dictionary
        res[f"peak_{idx_peak + 1}"]["fit_mu"]          = mu_fit
        res[f"peak_{idx_peak + 1}"]["fit_sigma"]       = sigma_fit
        res[f"peak_{idx_peak + 1}"]["fit_chi_squared"] = chi_squared

        # Normalizza l'istogramma e la curva gaussiana
        hist /= np.sum(hist)
        bell_curve /= np.sum(bell_curve)

        # Plot istogramma e fit
        ax1.bar(bin_edges[:-1], hist, width=bin_size1D, alpha=0.5, 
                color=hist1D_colors[idx_peak], label='Data')
        ax1.plot(bin_edges[:-1], bell_curve, color='red', label='Gaussian Fit')
        # Annotate plot with results
        ax1.plot([], [], color='white', label=f'Mean: {mu_fit:.5f}\nSigma: {sigma_fit:.5f}\nChi^2: {chi_squared:.5f}')
        
        if logscale:
            ax1.set_yscale('log')
            min_value = np.min(hist[hist > 0])
            ax1.set_ylim([10 ** np.floor(np.log10(min_value)), 1.1])
        
        ax1.set_xlabel('Relative Difference: (y_real - y_pred) / y_real')
        ax1.set_ylabel('Normalized Counts')
        ax1.set_title('Relative Difference Histogram & Gaussian Fit')
        ax1.legend()

        ### --- SUBPLOT 2: ISTOGRAMMA 2D DIFFERENZE ASSOLUTE --- ###
        ax2 = axes[1]
                
        # Calcola il range per i rapporti e le aree reali
        x_min, x_max = np.min(abs_diff[:, idx_peak]), abs_diff[:, idx_peak].max()  # Limiti fissi per il rapporto
        # x_min, x_max = -1000, 1000  # Limiti fissi per il rapporto
        y_min, y_max = np.min(y_real), np.max(y_real)
        
        # Determina i bin
        if bin_size2D_x is None:
            bin_size2D_x = (x_max - x_min) / 100  # Default: 100 bins
        if bin_size2D_y is None:
            bin_size2D_y = (y_max - y_min) / 100  # Default: 100 bins
        # Calcolo dei bin edges
        x_bins = np.arange(x_min, x_max + bin_size2D_x, bin_size2D_x)
        y_bins = np.arange(y_min, y_max + bin_size2D_y, bin_size2D_y)
        
        counts, _, _ = np.histogram2d(abs_diff[:, idx_peak].flatten(), 
                                      y_real[:, idx_peak].flatten(), 
                                      bins=[x_bins, y_bins])
        
        if logscale:
            counts = np.log1p(counts)

        img = ax2.imshow(counts.T, origin='lower', 
                         cmap=hist2D_colors[idx_peak],
                         extent=[x_min, x_max, y_min, y_max],
                         aspect='auto')
        plt.colorbar(img, ax=ax2, label='Counts (log scale)' if logscale else 'Counts')

        ax2.set_xlabel('Absolute difference (y_real - y_pred)')
        ax2.set_ylabel('Real Values (y_real)')
        ax2.set_title('2D Histogram of Absolute Differences')

        ### --- SALVATAGGIO E VISUALIZZAZIONE --- ###
        plt.suptitle(f'{title} {idx_peak+1}° peak')
        
        if path:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, 'combined_histograms.png'))
        
        plt.show()
    return res

##############################################################################################################################

def eval_method(y_real, y_pred, method, path=None, lim=1000, res=None):
    res_dict = {"MRR": mean_difference_relative_ratio(y_real, y_pred)}
    print(f'Mean Relative Ratio {res_dict["MRR"]}')
    # plot__difference_relative_ratio(y_real, y_pred, bin_size=0.001, ylogscale=False, path=path,
    #                                 title=f'difference_relative_ratio {method}')
    plot__difference_relative_ratio(y_real, y_pred, bin_size=0.001, ylogscale=True, path=path,
                                    title=f'difference_relative_ratio {method}')
    plot_difference_relative_ratio_with_percentiles(y_real, y_pred, bin_size=0.001, ylogscale=True, path=path,
                                                    title=f'difference_relative_ratio {method}')
    print__false_prediction_rate(y_real, y_pred, [0.01, 0.1, 0.2], res=res_dict)
    print__difference_over_mean(y_real, y_pred, res=res_dict)
    # plot__2d_bars(y_real, y_pred, relative=True, logscale=True, path=path,
    #               x_min=-0.5, x_max=0.5, title=f'2D Relative Difference (logscale) {method}')
    # plot__2d_bars(y_real, y_pred, relative=False, logscale=False, path=path,
    #               x_min=-lim, x_max=lim, title=f'2D Absolute Difference {method}')
    # plot__2d_bars(y_real, y_pred, relative=False, logscale=True, path=path,
    #               x_min=-lim, x_max=lim, title=f'2D Absolute Difference (logscale) {method}')
    # plot__2d_bars_with_quartiles(y_real, y_pred, 
    #                             title=f'2D Histogram with Quartiles  {method}', 
    #                             path=path)
    # plot__hists(y_real, y_pred, new_max=1, new_min=-1, logscale=False, path=path, res=res_dict,
    #             title=f'Relative difference cutscale {method}')
    # plot__gaussian_fitted(y_real, y_pred, new_max=1, new_min=-1, logscale=False, path=path, res=res_dict,
    #                       title=f'Relative difference cutscale with gaussian fit {method}')
    plot__hists(y_real, y_pred, new_max=1, new_min=-1, logscale=True, path=path, res=res_dict,
                title=f'Relative difference cutscale (logscale) {method}')
    # plot__gaussian_fitted(y_real, y_pred, new_max=1, new_min=-1, logscale=True, path=path, res=res_dict,
    #                       title=f'Relative difference cutscale with gaussian fit (logscale) {method}')
    plot_combined_histograms(y_real, y_pred, new_max=1, new_min=-1, 
                             logscale=True, path=path, res=res_dict,
                             bin_size1D=0.01,
                             title=f'Histograms {method}')
    if res is None:
        res = {f"{method}": res_dict}
    else:
        res[f"{method}"] = res_dict
    return res

##############################################################################################################################




def create_comparison_table(data):
    """
    Crea una tabella di confronto da un dizionario di risultati.
    
    Args:
        data (dict): Dizionario contenente i risultati per diversi modelli e picchi.
    
    Returns:
        pd.DataFrame: Tabella di confronto organizzata per metrica, picco e modello.
    """
    # Lista per raccogliere i dati in formato tabellare
    rows = []

    # Itera attraverso i modelli nel dizionario
    for model_name, model_data in data.items():
        # Aggiungi la metrica globale 'MRR'
        rows.append({
            "Model": model_name,
            "Metric": "MRR",
            "Peak": "Global",
            "Value": model_data["MRR"]
        })
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