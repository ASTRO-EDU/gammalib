import json
from typing import Union, Dict, Any
from gammafunctions import GmPiecewiseExp, GmConvolution, GmConvolutionFOrd, GmORSA, GmSemiGaussian
from gammafunc import GammaFunc


class GammaSim:
    method_mapping = {
        1: GmPiecewiseExp,
        2: GmConvolution,
        3: GmConvolutionFOrd,
        4: GmORSA,
        5: GmSemiGaussian
    }

    def __new__(cls, config: Union[str, Dict[str, Any]]) -> GammaFunc:
        """
        Create an object GammaSim, a simulator for GAMMA-FLASH data from a configuration file or a dictionary.

        ## Args:
        * `config`: Path to a JSON configuration file (str) or a dictionary containing the configuration (dict).

        ## Returns:
        * `GammaFunc`: An instance of the appropriate method based on `wf_shape`.
        """
        if isinstance(config, dict):
            config_data = config  # Usa direttamente il dizionario
        elif isinstance(config, str):
            with open(config, 'r') as file:
                config_data = json.load(file)  # Carica il file JSON
        else:
            raise TypeError("config must be a dictionary or a valid JSON file path (str).")

        wf_shape = config_data["wf_shape"]
        method = cls.method_mapping.get(wf_shape)
        if method is None:
            raise ValueError(
                f"Invalid wf_shape: {wf_shape}. "
                f"Allowed values are: {list(cls.method_mapping.keys())}"
            )

        return method(config_data)  # Passa il dizionario invece del path