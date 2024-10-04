from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil 
from scipy.stats import norm
from scipy.optimize import curve_fit

def filter_goodarea(area_real, area_pred): 
  return area_real[np.logical_and(
                          area_pred[:,0] > 1,
                          area_pred[:,1] > 1)
                  ]


def arr(area_real, area_pred):
    """
    Area Relative Ratio:
    """
    return (area_real-area_pred) / area_real


def marr(area_real, area_pred):
    """
    Mean Area Relative Ratio:
    """
    return np.mean(np.abs((area_pred-area_real) / area_real))


def npreds_over_nreal(area_real, area_pred):
    """
    Number predictions over Number real:
    """
    # Filtra area_pred per includere solo gli elementi che hanno la stessa lunghezza degli elementi corrispondenti in area_real
    filtered_area_pred = [ap for ap, ar in zip(area_pred, area_real) if len(ap) == len(ar)]
    # Calcola il rapporto tra la lunghezza di filtered_area_pred e la lunghezza di area_real
    return len(filtered_area_pred) / len(area_real)


def plot_ARR(area_real, area_pred, bin_size=0.1, path=None, xlogscale=False):
    """
    Plots histograms and boxplot ARR and print the Mean ARR in the title
    """
    for idx_peak in range(area_real.shape[-1]):
        # Calcolo di MARR e ARR
        marr_value = marr(area_real=area_real[:,idx_peak], area_pred=area_pred[:,idx_peak])
        arr_values = arr(area_real=area_real[:,idx_peak], area_pred=area_pred[:,idx_peak])
        # Definisci il numero di bin e il range
        num_bins = min(ceil((arr_values.max() - arr_values.min()) / bin_size), 100)
        # raise Exception('Stop')
        # Creazione dei subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Calcola gli istogrammi
        hist1, bin_edges = np.histogram(arr_values, bins=num_bins, density=True)
        hist1 = hist1 / np.sum(hist1)
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][idx_peak] # Seleziona il colore
        # Plot degli istogrammi utilizzando plt.bar()
        axs[0].bar(bin_edges[:-1], hist1, width=bin_size, color=color, label=f'diff {idx_peak+1}° peak')
        axs[0].set_title(f'Histogram ARR {idx_peak+1}° peak')
        axs[0].set_yscale('log')
        if xlogscale:
          axs[0].set_xscale('symlog')
        axs[0].set_ylabel('normalized counts')
        axs[0].set_xlabel('ARRs')
        # Plot del boxplot
        axs[1].boxplot(arr_values)
        axs[1].set_title(f'Boxplot ARR {idx_peak+1}° peak')
        axs[1].set_yscale('symlog')
        # Aggiunta di MARR come testo nel subplot dell'istogramma
        fig.suptitle(f'MARR on {idx_peak+1}° peak: {marr_value:.5f}', fontsize=14)
        # Visualizzazione del plot
        plt.tight_layout()
        if not path is None:
            # Crea ricorsivamente il percorso se non esiste
            os.makedirs(path, exist_ok=True)
            # Salva la figura nella cartella desiderata
            plt.savefig(os.path.join(path, f'ARR_plot__{idx_peak+1}.png'))
        plt.show()



def false_intpred_rate(area_real, area_pred, alpha: float = 0.5):
    """
    False Area Prediction Relative Rate:
    """
    arr = np.abs((area_pred-area_real) / area_real)
    fpred_len = len(arr[arr > alpha])
    return fpred_len/len(arr)



def print_faprr(area_real, area_pred, alpha_list: List):
    for idx_peak in range(area_real.shape[-1]):
        for alpha in alpha_list:
            print(f'False Prediction Rate Integral on {idx_peak+1}° peak (alpha={alpha}): '
                  f'{false_intpred_rate(area_real=area_real[:,idx_peak], area_pred=area_pred[:,idx_peak], alpha=alpha):.5f}')
        print()



def areaovermean_error(area_real, area_pred):
    """
    Area Difference Over Mean Real Area percentage error
    """
    diff = np.abs(area_pred - area_real)
    return np.mean(diff)/np.mean(area_real)*100



def print_aome(area_real, area_pred):
    """
    print Area Difference Over Mean Real Area percentage error
    """
    for idx_peak in range(area_real.shape[-1]):
        print(f'Area over mean on {idx_peak+1}° peak: '
              f'{areaovermean_error(area_real=area_real[:,idx_peak], area_pred=area_pred[:,idx_peak]):.5f}')



def plot_hists(area_real, area_pred,
               title='Diff_relative_cutscl', new_max=1, new_min=-1, bin_size=0.01, logscale=False, path=None):
    # Calcola la distanza fra area reale e predetta dividendo per l'area reale
    diff = (area_real - area_pred) / area_real
    # Tutti gli outliers sono raggruppati nello stesso estremo
    diff[diff > 1.] = 1.
    diff[diff < -1.] = -1.

    # Definisci il numero di bin e il range
    num_bins = ceil((new_max-new_min)/bin_size)
    range_min = new_min
    range_max = new_max
    print(f'### {title}')
    diff_mean   = diff.mean(axis=0)
    diff_median = np.median(diff, axis=0)
    diff_std    = diff.std(axis=0)
    print(f'{title} mean: {diff_mean}, std: {diff_std}')
    for idx_peak in range(diff.shape[-1]):
        # Calcola gli istogrammi
        hist1, bin_edges = np.histogram(diff[:,idx_peak], bins=num_bins, range=(range_min, range_max), density=True)
        hist1 = hist1 / np.sum(hist1)
        # Plot degli istogrammi utilizzando plt.bar()
        plt.bar(bin_edges[:-1], hist1, width=bin_size, alpha=0.5, label=f'diff {idx_peak+1}° peak')
        if logscale:
            plt.yscale('log')
        # Aggiungi etichette per la media dei due istogrammi
        plt.text(diff_mean[idx_peak] - 0.1,
                 np.max(hist1)/(idx_peak+1), f'μ_{idx_peak+1}: {diff_mean[idx_peak]:.5f},\nσ_{idx_peak+1}: {diff_std[idx_peak]:.5f}',
                 fontsize=10, verticalalignment='top',
                 horizontalalignment='right')
    plt.legend()
    plt.xlabel('diff')
    plt.ylabel('counts normalized')
    plt.title(f'{title} tra real e reco (bin_size={bin_size})')
    if not path is None:
        # Crea ricorsivamente il percorso se non esiste
        os.makedirs(path, exist_ok=True)
        # Salva la figura nella cartella desiderata
        plt.savefig(os.path.join(path, f'diff_relative.png'))
    plt.show()



def plot_gaussian_fitted(area_real, area_pred, 
                         title='Diff_relative_cutscl vs Gaussian fit', new_max=1, new_min=-1, bin_size=0.01, logscale=False, path=None):
    ###################################
    # Fit della distribuzione gaussiana
    def gaussian_fit(x, mu, sigma):
        return norm.pdf(x, mu, sigma)
    ###################################
    # Calcola la distanza fra area reale e predetta dividendo per l'area reale
    diff = (area_real - area_pred) / area_real
    # Tutti gli outliers sono raggruppati nello stesso estremo
    diff[diff > 1.] = 1.
    diff[diff < -1.] = -1.

    # Definisci il numero di bin e il range
    num_bins = ceil((new_max-new_min)/bin_size)
    range_min = new_min
    # range_min = new_min
    range_max = new_max
    # range_max = new_max
    res = []
    for idx_peak in range(diff.shape[-1]):
        print(f'{title} {idx_peak+1}° peak')
        # Dati per il fitting
        data = diff[:, idx_peak]  # Sostituisci con i tuoi dati
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][idx_peak] # Seleziona il colore
        # Stima iniziale dei parametri
        initial_guess = [np.mean(data), np.std(data)]
        # Fitting dei dati
        hist, bin_edges = np.histogram(data, bins=num_bins, range=(range_min, range_max), density=True)
        params, covariance = curve_fit(gaussian_fit, bin_edges[:-1], hist, p0=initial_guess)
        # Estrai media e sigma
        mu_fit, sigma_fit = params
        bell_curve = gaussian_fit(bin_edges[:-1], mu_fit, sigma_fit)
        # Make hist and gaussian fit a probability in [0, 1]
        hist = hist / np.sum(hist)
        bell_curve = bell_curve / np.sum(bell_curve)
        # Calcolo del chi quadrato
        epsilon = 1e-10
        chi_squared = np.sum(((bell_curve - hist) ** 2) / (hist + epsilon))
        # Stampa media, sigma e chi quadrato
        print(f"Mean: {mu_fit}")
        print(f"Sigma: {sigma_fit}")
        print(f"Chi^2: {chi_squared}")
        res.append([mu_fit, sigma_fit, chi_squared])

        # Plot dell'istogramma e della distribuzione gaussiana adattata
        plt.bar(bin_edges[:-1], hist, width=bin_size, alpha=0.5, label='Data', color=color)
        plt.plot(bin_edges[:-1], bell_curve, color='red', label='Gaussian Fit')
        # print(bell_curve)
        if logscale:
            plt.yscale('log')
            minvalue = np.min(hist[hist>0])
            log_minimo = np.log10(minvalue)
            potenza_intera = np.floor(log_minimo)
            potenza_di_10 = 10 ** potenza_intera
            plt.ylim([potenza_di_10, 1])

        # Aggiunta delle etichette per media, sigma e chi quadrato
        # plt.text(mu_fit + 0.1 if mu_fit <= 0 else mu_fit - 0.1,
        plt.text(mu_fit - 0.1,
                np.max(hist)*0.8, f'Mean: {mu_fit:.5f}\n'
                                  f'Sigma: {sigma_fit:.5f}\n'
                                  f'Chi^2: {chi_squared:.5f}\n', fontsize=10,
                verticalalignment='bottom',
                # horizontalalignment='left' if mu_fit <= 0 else 'right')
                horizontalalignment='right')
        plt.legend()

        plt.xlabel('diff')
        plt.ylabel('counts normalized')
        plt.title(f'{title} tra real e reco (bin_size={bin_size})')
        if not path is None:
            # Crea ricorsivamente il percorso se non esiste
            os.makedirs(path, exist_ok=True)
            # Salva la figura nella cartella desiderata
            plt.savefig(os.path.join(path, f'diff_relative_Gfit__{idx_peak+1}.png'))
        plt.show()
    return res
