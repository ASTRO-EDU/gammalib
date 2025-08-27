from configuration.model.config_model import load_config
import sys
print(sys.path)
sys.path.append("mnt/Windows/SHARED/ws_vscode/gammalib")
sys.path.append("/mnt/Windows/SHARED/ws_vscode/gammalib/area_eval/")
from gammasim import GammaSim
from implementation.algorithm import TrapezoidalShaperAlg
from implementation.filt_param_handlers import read_trap_params

 # Carica la configurazione
config_file_name = "config_tps_config_method2-no-noise.json"
cfg = load_config(config_file_name)
print(f"Configuration imported! \n {cfg} \n\n")

# Istanzio il simulatore GammaSim
config_file = cfg.gammasim_cfg
saturation = False
gammasim = GammaSim(config_file)
gammasim.generate_dataset(saturation)

# Crea l'istanza dell'algoritmo e calcola i risultati
ds = gammasim.get_dataset()
M = gammasim.get_params()[0][0]['tau2']
sampling_time=gammasim.get_sampling_time()
heights=gammasim.get_heights()
alg = TrapezoidalShaperAlg(dataset=ds, config=cfg, sampling_time=sampling_time, M=M)

csv_file="notebook_values.cfg"
trap_heights_before = alg.find_gain(heights, out=csv_file)
print("Applying scaling")
alg.set_mean_computed_scaling_from_file(csv=csv_file)
trap_heights_after = alg.find_gain(heights)

if trap_heights_after > 0.9 and trap_heights_after < 1.1:
    print("Trap scaling successful!!!")
print(read_trap_params(csv_file))