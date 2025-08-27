# config_model.py
import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

class TimeFilterParams(BaseModel):
    alpha_l: float = Field(..., description="Low pass filter coefficient")
    alpha_h: float = Field(..., description="High pass filter coefficient")
    gain_k: float = Field(..., description="Filter gain")
    in_cond: List[float] = Field(..., description="Initial conditions")
    th_dy: Optional[float] = Field(None, description="Trigger threshold for dy signal")
    th_d2y: Optional[float] = Field(None, description="Trigger threshold for d2y signal")
    dev_ord_det: int = Field(..., description="Order of time derivated signal")
    zero_cr_window: int = Field(..., description="n samples where to look for zero cross after candidate peak")
    window_sz: int = Field(..., description="Smoothing window for thd2y")

class TrapFiltParams(BaseModel):
    m: int = Field(..., description="Flat top (samples)")
    l: int = Field(..., description="Slope (samples)")
    ftd_s: int = Field(..., description="Flat top delay to start (samples). height_window_start = trap_flat_start+ftd_s")
    ftd_e: int = Field(..., description="Flat top delay before end (samples). height_window_end = trap_flat_end-ftd_e")
    
class ChargeIntParams(BaseModel):
    predelay: int = Field(..., description="Samples to integrate signal before tstart. t_start_integration = t_start-predelay")
    width: int = Field(..., description="Samples to end integrate signal. t_stop_integration = t_start-predelay+width")

class Config(BaseModel):
    gammasim_cfg: Path = Field(..., description="Path to configuration file for simulator")
    time_filter: TimeFilterParams
    trap_filter: TrapFiltParams
    charge_int_filter: ChargeIntParams

# Funzione per caricare la configurazione da un file JSON
def load_config(file_name: str) -> Config:
    # Ottieni il percorso della directory parente dello script
    file_name      = os.path.basename(file_name)
    print(file_name)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Ottieni la directory parente
    file_path = os.path.join(parent_dir, file_name)
    print(file_name)

    # Leggi il file JSON e carica i dati
    with open(file_path, "r") as f:
        data = json.load(f)

    # Ricrea un'istanza della configurazione utilizzando Pydantic
    config = Config(**data)
    return config


cfg_default = Config(
        gammasim_cfg="",
        time_filter=TimeFilterParams(
            alpha_l=0.002,
            alpha_h=0.9,
            gain_k=10e-5,
            in_cond=[0],
            th_dy=None,
            th_d2y=None,
            dev_ord_det=1,
            window_sz=15,
            zero_cr_window=10
        ),
        trap_filter=TrapFiltParams(
        m=60,
        l=40,
        ftd_s=20,
        ftd_e=0
        ),
        charge_int_filter=ChargeIntParams(
            predelay=100,
            width=400
        )
    )