# __Gamma-FLASH simulator__

## Setup
Create a new virtual environment
```
python -m venv gammasim_venv
```
After activated it you can install all the dependencies.
```
source ./gammasim_venv/bin/activate
pip install -r requirements.txt
```

## How to use the simulator

The simulator for GammaSky can be used by importing the `GammaSim` class as explained below
```
from gammasim import GammaSim
```
Subsequently, to instantiate the GammaSim object you need to pass the path to the simulator configuration file as an argument
```
gammasim = GammaSim('config.json')
```

The configuration file is a json file that is made up of the following keys:
* `size`: is the dimension of the dataset, the number of data samples
* `xlen`: is the length of the time series
* `maxcount_value`: it is the maximum measurable count in cases of saturation
* `max_peaks`: is the number of peaks present in each waveform
* `gauss_maxrate`: is a value between 0 and 1, and represents the percentage of mean and std deviation with respect to the `maxcount_value` and is used in the case where `gauss_std` or `gauss_mean` have been set to `none` to calculate the standard deviation or the average, respectively, for the Gaussian noise to be added to the clean signal
* `gauss_std`: standard deviation for the Gaussian noise to be added to the clean signal. When it is `none`, `maxcount_value * gauss_maxrate` is used as standard deviation
* `gauss_mean`: mean for the Gaussian noise to be added to the clean signal. When it is `none`, `maxcount_value * gauss_maxrate` is used as mean
* `bkgbase_level`: is the constant levelwhere is located the background noise. It is not raleted to the Gaussian noise
* `gamma_min_wtSat`: it is the minimum value that can be assumed by the first peaks in the dataset in case of saturation
* `gamma_max_wtSat`: it is the maximum value that can be assumed by the first peaks in the dataset in case of saturation
* `gamma_min_noSat`: it is the minimum value that can be assumed by the first peaks in the dataset in case of absense of saturation
* `gamma_max_noSat`: it is the maximum value that can be assumed by the first peaks in the dataset in case of absense of saturation
* `tstart_min`: it is the minimum value that can be assumed by the time start index in the dataset 
* `tstart_max`: it is the maximum value that can be assumed by the time start index in the dataset 
* `delta_tstart`: it is the minimum distance that seperate two different peaks
* `tau1_min`: it is the minimum time bins necessary to reach the peak 
* `tau1_max`: it is the maximum time bins necessary to reach the peak 
* `tau2_min`: it is the minimum time bins necessary to reach the background level after the peak 
* `tau2_max`: it is the maximum time bins necessary to reach the background level after the peak

## Methods

* > `generate_dataset(saturation)`

    Once you defined the config file you can generate your dataset as follows:
    ```
    gammasim.generate_dataset(saturation)
    100%|██████████| 1000/1000 [00:00<00:00, 4453.80it/s]
    ```
    the `saturation` argument is used to define the desire to reach or not the saturation level given by the `maxcount_value`

* > `plot_wf(idx)`

    PLots the `idx`-th waveform. If `idx` is equal to `max` (`min`) it will plot the waveforms with the maximum (minimum) area label. Its default value is `random`, in which case it will plot a waveform at random from the dataset

* > `get_dataset()`

    Returns the dataset of waveforms

* > `def get_labels()`

    Returns the dataset of waveforms without Gaussian noise
    
* > `def get_labelsSplit()`

    Returns `size * xlen * max_peaks` waveforms without Gaussian noise and without background
    
* > `def get_areas()`

    Returns an array of dimension `size * max_peaks` of constant integer values representing the area of each peaks 