from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union
from typing_extensions import Annotated

class ConfigModel(BaseModel):
    size: Annotated[int, Field(ge=1)]  # Deve essere almeno 1
    xlen: Annotated[int, Field(ge=1)]  # Deve essere almeno 1
    sampling_time: Annotated[float, Field(ge=1e-9)]  # Intervallo di tempo positivo
    n_bit_quantization: Annotated[int, Field(ge=1)]  # Numero di bit di quantizzazione positivo
    maxcount_value: Annotated[int, Field(ge=0)]  # Valore massimo non negativo
    mincount_value: Annotated[int, Field(ge=0)]  # Valore minimo non negativo
    max_peaks: Annotated[int, Field(ge=0)]  # Numero di picchi non negativo
    gauss_maxrate: Annotated[float, Field(ge=0.0, le=1.0)]  # Valore tra 0 e 1
    gauss_std: Optional[Union[float, str]] = "none"  # Può essere un numero o "none"
    gauss_mean: Optional[Union[float, str]] = "none"  # Può essere un numero o "none"
    bkgbase_level: Annotated[int, Field(ge=0)]  # Livello del rumore
    gamma_min_wtSat: Annotated[int, Field()]  # Minimo valore per il picco in saturazione
    gamma_max_wtSat: Annotated[int, Field()]  # Massimo valore per il picco in saturazione
    gamma_min_noSat: Annotated[int, Field()]  # Minimo valore per il picco senza saturazione
    gamma_max_noSat: Annotated[int, Field()]  # Massimo valore per il picco senza saturazione
    tstart_min: Annotated[int, Field(ge=0)]  # Minimo valore per l'inizio del tempo
    tstart_max: Annotated[int, Field(ge=0)]  # Massimo valore per l'inizio del tempo
    delta_tstart: Annotated[int, Field(ge=0)]  # Distanza minima tra i picchi
    wf_shape: Annotated[int, Field(ge=0)]  # Forma del segnale (es. 1 per gaussiana)
    tau1_min: Annotated[Union[int, float], Field(ge=0)]  # Minimo tempo per raggiungere il picco
    tau1_max: Annotated[Union[int, float], Field(ge=0)]  # Massimo tempo per raggiungere il picco
    tau2_min: Annotated[Union[int, float], Field(ge=0)]  # Minimo tempo per raggiungere il livello di fondo
    tau2_max: Annotated[Union[int, float], Field(ge=0)]  # Massimo tempo per raggiungere il livello di fondo
    gauss_kernel_min: Annotated[float, Field(ge=0)]  # Kernel Gaussiano minimo
    gauss_kernel_max: Annotated[float, Field(ge=0)]  # Kernel Gaussiano massimo

    # Validazione per gauss_std e gauss_mean: se sono "none", usiamo `gauss_maxrate` per il calcolo
    @field_validator('gauss_std', 'gauss_mean', mode='before')
    def validate_gauss_fields(cls, v, field):
        if v == "none":
            return None  # Puoi rimuovere questa linea o adattarla a un valore di default
        return v

    class Config:
        str_min_length = 1  # Le stringhe devono essere almeno di 1 carattere
