# trapezoidal_shaper

clone gammalib
```
$ git clone https://github.com/ASTRO-EDU/gammalib.git
```

follow the instructions to build virtualenv gammasim

install additional deps 
```
$ pip install -r requirements.txt
```

## Usage

activate gammasim env and export PYTHONPATH with gammasim path:
```
$ source /path/to/gammasim_env/bin/activate
(gammasim) $ export PYTHONPATH=/path/to/gammalib/gammasim
```

## visualize waveform model 

https://www.desmos.com/calculator/u1dtk6dqgx

## trapezoidal shaper reference:
https://drive.google.com/drive/folders/1kx1BpFW2vLYtW8lw7NBKV7afeeUhIyzG

## vscode configuration:
{
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "</path/to>/gammalib/gammasim/:</path/to>/gammalib/area_eval/"
    },
    "python.analysis.extraPaths": [
        "</path/to>/gammalib/gammasim/",
        "</path/to>/gammalib/area_eval/"
    ]
}