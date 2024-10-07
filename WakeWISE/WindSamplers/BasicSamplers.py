from csv import DictReader, DictWriter
from pathlib import Path

import numpy as np
from scipy.stats import weibull_min

from WakeWISE.utils.time import TimeKeeper
from WakeWISE.WindSamplers.WindSampler import WindSampler

base_path = Path(__file__).parent

class UniformSampler(WindSampler):

    def __init__(self, params: dict, timestep: int, time: TimeKeeper) -> None:

        assert {'max_wind_speed'}.issubset(params), f'Missing parameters for {self.__class__.__name__}'

        self.max_wind_speed = params['max_wind_speed']

        self.time = time

    def step(self) -> dict:

        next_ws = np.random.uniform(0, self.max_wind_speed)
        next_wd = np.random.uniform(0, 360)
        ti = np.random.uniform(0, 0.5)

        # Just to ensure we are keeping within limits...
        next_ws = np.clip(next_ws, 0, 30)
        ti = np.clip(ti, 0, 0.5)

        info = {
            'wd': next_wd,
            'ws': next_ws,
            'ti': ti,
            'vector': np.array([next_wd, next_ws, ti]),
            'other': {}
        }
        
        return info
    
    def reset(self, time: TimeKeeper) -> dict:

        return self.step(time)
    
class WeibullSampler(WindSampler):

    def __init__(self, scale: float, shape: float) -> None:

        self.scale = scale
        self.shape = shape

        self.alpha_min_limit = -0.3
        self.alpha_max_limit = 2.5

    def step(self, time: TimeKeeper) -> dict:

        next_ws = np.random.weibull(self.shape) * self.scale
        next_wd = np.random.uniform(0, 360)
        ti = np.random.uniform(0, 0.5)

        alpha_min = max(0.15 - 0.23 * (30 / next_ws) * (1 - (0.4 * np.log(130 / 110)**2)), self.alpha_min_limit)
        alpha_max = min(0.22 + 0.4 * (130 / 110) * (30 / next_ws), self.alpha_max_limit)
        alpha = np.random.uniform(0, 1) * (alpha_max - alpha_min) + alpha_min

        # Just to ensure we are keeping within limits...
        next_ws = np.clip(next_ws, 0, 30)
        ti = np.clip(ti, 0, 0.5)

        info = {
            'wd': next_wd,
            'ws': next_ws,
            'ti': ti,
            'alpha': alpha,
            'vector': np.array([next_wd, next_ws, ti, alpha]),
            'other': {}
        }
        
        return info
    
    def reset(self, time: TimeKeeper) -> dict:

        return self.step(time)
    
class AnholtBasicWindSampler(WindSampler):
    """
    Basic wind sampler using data inferred from the Anholt wind farm
    The wind direction is binned into 18 discrete states.
    Note that the first bin goes from -10 to 10 degrees.
    The direction modelled by each bin is the middle of the bin.

    Each time step, the wind bin is sampled from a multinomial distribution
    The probability of each wind bin is given by a 18x1 array, with the probabilities summing to 1
    The wind speed is then sampled from a weibull distribution for the given wind bin
    """

    def __init__(self, params: dict) -> None:
        """
        Initializes the wind sampler with the given parameters

        Args:
            params (dict): the parameters for the wind sampler
        """

        # Default case: load the data from the file
        if not {'p_windcases', 'weibull_parameters'}.issubset(params):
            self.p_wind_state = np.load(f'{base_path}/data/AnholtV1/10min/wind_bin_probs_SIMPLE.npy')
            self.weibull_parameters = np.load(f'{base_path}/data/AnholtV1/10min/weibull_params_SIMPLE.npy')

        # Custom case: load the data from the given parameters
        else:
            try:
                self.p_wind_state = np.array(params['p_transition'])
            except Exception:
                raise ValueError(f'Invalid wind state probability array for {self.__class__.__name__}.')
            
            try:
                self.weibull_parameters = np.array(params['weibull_parameters'])
            except Exception:
                raise ValueError(f'Invalid weibull parameters for {self.__class__.__name__}.')
        
        # p_wind_state is of shape Nx1, where there are N wind states each with a probability
        # weibull_parameters is of shape Nx3, where there are N wind states with three parameters for each state
        assert self.p_wind_state.ndim == 1, f'Invalid wind state probability array for {self.__class__.__name__}: too many probabilities per wind state'
        assert self.p_wind_state.shape[0] == self.weibull_parameters.shape[0], f'Invalid wind state probability array for {self.__class__.__name__}: not matching weibull parameters'
        assert self.weibull_parameters.shape[1] == 3, f'Invalid weibull parameters for {self.__class__.__name__}: not 3 parameters per state; required for weibull_min'
        assert np.isclose(np.sum(self.p_wind_state), 1), f'Invalid wind state probability array for {self.__class__.__name__}: not summing to 1'
        assert np.all(self.p_wind_state >= 0), f'Invalid wind state probability array for {self.__class__.__name__}: not all non-negative'

        self.alpha_min_limit = -0.3
        self.alpha_max_limit = 2.5

    def step(self) -> dict:

        # Wind speed and direction are sampled from the joint distribution of wind speeds and directions for each month
        # The joint distribution is a multinomial distribution, with the probabilities of each wind state given by the transition matrix
        # The wind speed and direction are then sampled from the weibull distribution for the given wind state
        # The wind state is determined by the previous wind state and the transition matrix
        next_wd = np.random.multinomial(1, self.p_wind_state).argmax()
        shape, loc, scale = self.weibull_parameters[next_wd, :]
        next_ws = weibull_min.rvs(shape, loc=loc, scale=scale)

        ti_max = np.clip(0.18 * (0.75 + 5.6 / next_ws), None, 0.5)
        ti_min = 0.04
        ti = np.random.uniform(0, 1) * (ti_max - ti_min) + ti_min

        alpha_min = max(0.15 - 0.23 * (30 / next_ws) * (1 - (0.4 * np.log(130 / 110)**2)), self.alpha_min_limit)
        alpha_max = min(0.22 + 0.4 * (130 / 110) * (30 / next_ws), self.alpha_max_limit)
        alpha = np.random.uniform(0, 1) * (alpha_max - alpha_min) + alpha_min

        # Just to ensure we are keeping within limits...
        next_ws = np.clip(next_ws, 0, 30)
        ti = np.clip(ti, 0, 0.5)

        info = {
            'wd': next_wd,
            'ws': next_ws,
            'ti': ti,
            'alpha': alpha,
            'vector': np.array([next_wd, next_ws, ti, alpha]),
            'other': {}
        }

        return info

    def reset(self) -> dict:

        return self.step()

################################################################################################################
# TAKEN FROM: https://github.com/AlgTUDelft/wind-farm-env/blob/main/wind_farm_gym/wind_process/mvou_process.py #
# Gregory Neustroev, Sytze P.E. Andringa, Remco A. Verzijlbergh, Mathijs M. de Weerdt, 2022.                   #
################################################################################################################
class CSVSampler(WindSampler):
    """
    CSVProcess is a wind process that reads data from a `.csv`-file. The file must have the following format:

    First line contains the names of the atmospheric measurements, e.g., wind_speed or wind_direction
    The following lines contain the corresponding measurements. Each time `step` is called, the data from the next
    unused line is returned
    """

    def __init__(self, file):
        super().__init__()
        assert file is not None, 'a data file must be provided'
        assert isinstance(file, str), 'file name must be a string'

        with open(file, 'r') as input_file:
            dict_reader = DictReader(input_file)
            data = [{k: float(v) for k, v in line.items()} for line in dict_reader]
        self._data = data
        self._t = 0

    def save(self, file):
        keys = self._data[0].keys() if len(self._data) > 0 else []
        with open(file, 'w') as output_file:
            dict_writer = DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self._data)

    def step(self):
        item = self._data[self._t]
        self._t = (self._t + 1) % len(self._data)  # if there are no more lines in the data, start from the beginning
        return item

    def reset(self):
        self._t = 0