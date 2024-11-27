from configuration.model.config_model import load_config
import sys
print(sys.path)
sys.path.append("mnt/Windows/SHARED/ws_vscode/gammalib")
sys.path.append("/mnt/Windows/SHARED/ws_vscode/gammalib/area_eval/")
from gammasim import GammaSim
from implementation.algorithm import TrapezoidalShaperAlg
from implementation.filt_param_handlers import read_trap_params
import area_eval

 # Carica la configurazione
config_file_name = "config_tps_config_method2-w-noise.json"
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
alg = TrapezoidalShaperAlg(dataset=ds, config=cfg, sampling_time=sampling_time, M=M)

csv_file="notebook_values.cfg"
print(read_trap_params(csv_file))
alg.set_mean_computed_scaling_from_file(csv=csv_file)
alg.compute_all(plot=True)
a_output=alg.trap_heights_data
save_output_fig = "plot_eval"

area_eval.plot_ARR(area_pred=a_output, area_real=gammasim.get_areas(),path=save_output_fig)
area_eval.plot_hists(area_pred=a_output, area_real=gammasim.get_areas(),path=save_output_fig)
area_eval.plot_gaussian_fitted(area_pred=a_output, area_real=gammasim.get_areas(),path=save_output_fig)

