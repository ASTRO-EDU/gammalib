import json
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

    def __new__(self, config_path: str) -> GammaFunc:
        """
        Create an object GammaSim, a simulator for GAMMA-FLASH data from a configuration file.
        ## Args
        * `configfile_path`: configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as file:
            config = json.load(file)
        self.wf_shape = config["wf_shape"]
        self.method = self.method_mapping.get(self.wf_shape)
        if self.method is None:
            raise ValueError(
                f"Invalid wf_shape: {self.wf_shape}. "
                f"Allowed values are: {list(self.method_mapping.keys())}"
            )
        return self.method(config_path)