import math
import os
from csv import DictWriter
from pathlib import Path

import numpy as np
from gymnasium.utils import seeding
from numpy.random import uniform
from scipy.linalg import expm, pinv
from scipy.stats import norm, weibull_min

from WakeWISE.utils.logger import warn
from WakeWISE.utils.time import TimeKeeper
from WakeWISE.WindSamplers.BasicSamplers import CSVSampler
from WakeWISE.WindSamplers.WindSampler import WindSampler

base_path = Path(__file__).parent


class AnholtDetailedWindSampler(WindSampler):
    """
    Detailed wind sampler using data inferred from the Anholt wind farm
    The wind direction is binned into 18 discrete states.
    Note that the first bin goes from -10 to 10 degrees.
    The direction modelled by each bin is the middle of the bin.

    The change between the bins is modelled by a Markov Chain, with the transition matrix inferred from the data.
    This transition matrix is modelled for each month, and the wind speed is modelled by a Weibull distribution for each wind state.
    """


    def __init__(self, params: dict, timestep: int, time: TimeKeeper):

        assert timestep in [10], f'{self.__class__.__name__} does not currently support timesteps of {timestep}m'

        #  Default case: load the data from the file
        if not {'p_transition', 'weibull_parameters'}.issubset(params):
            self.p_transition = np.load(f'{base_path}/data/AnholtV1/{timestep}min/transition_matrices_monthly.npy')
            self.weibull_parameters = np.load(f'{base_path}/data/AnholtV1/{timestep}min/weibull_params.npy')

        # Custom case: load the data from the parameters
        else:
            try:
                self.p_transition = np.array(params['p_transition'])
            except Exception:
                raise ValueError(f'Invalid transition matrix for {self.__class__.__name__}.')
            
            try:
                self.weibull_parameters = np.array(params['weibull_parameters'])
            except Exception:
                raise ValueError(f'Invalid weibull parameters for {self.__class__.__name__}.')
        
        # p_transition is of shape NxNxM, where there are N wind states and M months
        # weibull_parameters is of shape Nx3xM, where there are N wind states and M months, with three parameters for each state
        assert self.p_transition.shape[0] == self.p_transition.shape[1], f'Invalid transition matrix for {self.__class__.__name__}: not square'
        assert self.p_transition.shape[0] == self.weibull_parameters.shape[0], f'Invalid transition matrix for {self.__class__.__name__}: not matching weibull parameters'
        assert self.weibull_parameters.shape[1] == 3, f'Invalid weibull parameters for {self.__class__.__name__}: not 3 parameters per state; required for weibull_min'
        assert self.p_transition.shape[2] == self.weibull_parameters.shape[2], f'Invalid transition matrix for {self.__class__.__name__}: not matching months'
        assert np.all(np.isclose(np.sum(self.p_transition, axis=1), 1)), f'Invalid transition matrix for {self.__class__.__name__}: not summing to 1'
        assert np.all(self.p_transition >= 0), f'Invalid transition matrix for {self.__class__.__name__}: not all non-negative'
        assert self.p_transition.shape[2] == 12, f'Invalid transition matrix for {self.__class__.__name__}: not 12 months'

        self.alpha_min_limit = -0.3
        self.alpha_max_limit = 2.5

        # Determine the equillibrium state for the Markov Chain
        # This is used to determine the initial state
        self.p_equillibrium = self._determine_equillibrium()

        self.time = time
        
        # Initialise the model
        self.reset()

    def step(self) -> tuple[float, float]:
        """
        Samples the wind speed and direction for the given time

        Args:
            time (TimeKeeper): the time for which to sample the wind

        Returns:
            tuple[float, float, float]: the wind speed, direction and turbulence intensity
        """

        # Determine the month
        month = self.time.months

        # The wind direction is modelled as a Markov Chain, with the transition matrix determined by the month
        # The wind speed is modelled by a Weibull distribution for the given wind state
        # First, we obtain this month's transition matrix and index the row corresponding to the previous wind state
        p_tr = self.p_transition[self.prev_wd, :, month]

        # Sample the next wind state according to the transition matrix
        next_wd = np.random.multinomial(1, p_tr).argmax()

        # Find the corresponding Weibull parameters for the wind speed
        shape, loc, scale = self.weibull_parameters[next_wd, :, month]

        # Sample the wind speed from the Weibull distribution
        next_ws = weibull_min.rvs(shape, loc=loc, scale=scale)

        # Update the previous wind state
        self.prev_wd = next_wd

        # Decode the wind direction
        next_wd *= 20

        # Add noise to the wind direction and speed
        next_wd += uniform(-10, 10)

        # Ensure wind direction wraps around
        next_wd %= 360
        
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

        # Return the wind speed and direction
        return info

    def reset(self) -> dict:
        """
        Reset the wind sampler to a new initial state.
        It takes the equillibrium state to sample the 'previous' wind state from
        """

        self.prev_wd = np.random.multinomial(1, self.p_equillibrium).argmax()

        return self.step()

    def _determine_equillibrium(self) -> np.ndarray:
        """
        Determines the equillibrium state for the Markov Chain, given the transition matrix
        """

        n_states = self.p_transition.shape[0]
        A = np.append(self.p_transition[:, :, 0].T - np.eye(n_states), np.ones(n_states).reshape(1, -1), axis=0)
        pinv = np.linalg.pinv(A)
        return pinv.T[-1]
    
class AnholtDetailedWindSamplerV2(WindSampler):
    """
    Detailed wind sampler using data inferred from the Anholt wind farm
    The wind direction is binned into 18 discrete states.
    Note that the first bin goes from -10 to 10 degrees.
    The direction modelled by each bin is the middle of the bin.

    The change between the bins is modelled by a Markov Chain, with the transition matrix inferred from the data.
    This transition matrix is modelled for each month.

    Different from the previous version, the wind speed is too modelled by a Markov Chain
    The transition matrix is dependent on both the current wind direction, AND the previous wind speed
    Wind speed is binned. To remove the discretization, uniform noise is added to the wind speed.
    """


    def __init__(self, params: dict, timestep: int, time: TimeKeeper) -> None:

        assert timestep in [10, 60, 1440], f'{self.__class__.__name__} does not currently support timesteps of {timestep}m'

        self.use_file = timestep if timestep != 1440 else 60

        #  Default case: load the data from the file
        if not {'p_transition_wd', 'p_transition_ws'}.issubset(params):
            self.p_transition_wd = np.load(f'{base_path}/data/AnholtV2/{self.use_file}min/{self.use_file}min_wd_transitions_monthly.npy')
            self.p_transition_ws = np.load(f'{base_path}/data/AnholtV2/{self.use_file}min/{self.use_file}min_ws_transitions.npy')

        # Custom case: load the data from the parameters
        else:
            try:
                self.p_transition_wd = np.array(params['p_transition_wd'])
            except Exception:
                raise ValueError(f'Invalid wind direction transition matrix for {self.__class__.__name__}.')
            
            try:
                self.p_transition_ws = np.array(params['p_transition_ws'])
            except Exception:
                raise ValueError(f'Invalid wind speed transition matrix for {self.__class__.__name__}.')
        
        # p_transition is of shape NxNxM, where there are N wind states and M months
        # weibull_parameters is of shape Nx3xM, where there are N wind states and M months, with three parameters for each state
        assert self.p_transition_wd.shape[0] == self.p_transition_wd.shape[1], f'Invalid wind direction transition matrix for {self.__class__.__name__}: not square'
        assert self.p_transition_wd.shape[0] == self.p_transition_ws.shape[0], f'Invalid wind direction transition matrix for {self.__class__.__name__}: not matching wind speed transition matrix'
        assert self.p_transition_ws.shape[1] == self.p_transition_ws.shape[1], f'Invalid wind speed transition matrix for {self.__class__.__name__}: not square'
        assert np.all(np.isclose(np.sum(self.p_transition_wd, axis=1), 1)), f'Invalid wind direction transition matrix for {self.__class__.__name__}: not summing to 1'
        assert np.all(np.isclose(np.sum(self.p_transition_ws, axis=2), 1)), f'Invalid wind speed transition matrix for {self.__class__.__name__}: not summing to 1'
        assert np.all(self.p_transition_wd >= 0), f'Invalid wind direction transition matrix for {self.__class__.__name__}: not all non-negative'
        assert np.all(self.p_transition_ws >= 0), f'Invalid wind speed transition matrix for {self.__class__.__name__}: not all non-negative'
        assert self.p_transition_wd.shape[2] == 12, f'Invalid wind direction transition matrix for {self.__class__.__name__}: not 12 months'

        self.alpha_min_limit = -0.3
        self.alpha_max_limit = 2.5

        # Determine the equillibrium state for the Markov Chain
        # This is used to determine the initial state
        self.p_equillibrium = self._determine_equillibrium()

        self.time = time
        
        # Initialise the model
        self.reset()

    def step(self) -> dict:
        """
        Samples the wind speed and direction for the given time

        Args:
            time (TimeKeeper): the time for which to sample the wind

        Returns:
            tuple[float, float, float]: the wind speed, direction and turbulence intensity
        """

        # Determine the month
        month = self.time.months

        # The wind direction is modelled as a Markov Chain, with the transition matrix determined by the month
        # The wind speed is modelled by a Weibull distribution for the given wind state
        # First, we obtain this month's transition matrix and index the row corresponding to the previous wind state
        p_tr_wd = self.p_transition_wd[self.prev_wd, :, month]

        # Sample the next wind state according to the transition matrix
        next_wd = np.random.multinomial(1, p_tr_wd).argmax()
        # next_wd = list(multinomial.rvs(1, p_tr_wd)).index(1)

        # The wind speed is modelled as a Markov Chain, with the transition matrix determined by
        # current wind direction and previous wind speed.
        # First, we obtain the row corresponding to this wind direction and previous wind speed
        p_tr_ws = self.p_transition_ws[next_wd, self.prev_ws, :]

        # Sample the next wind speed according to the transition matrix
        next_ws = np.random.multinomial(1, p_tr_ws).argmax()
        # next_ws = list(multinomial.rvs(1, p_tr_ws)).index(1)

        # Update the previous wind state
        self.prev_wd = next_wd
        self.prev_ws = next_ws

        # Decode the wind states
        next_wd *= 20
        next_ws += 0.5

        # Add noise to the wind direction and speed
        next_wd += uniform(-10, 10)
        next_ws += uniform(-0.5, 0.5)

        # Ensure wind direction wraps around
        next_wd %= 360

        ti_max = np.clip(0.18 * (0.75 + 5.6 / next_ws), None, 0.5)
        ti_min = 0.04
        ti = np.random.uniform(0, 1) * (ti_max - ti_min) + ti_min

        alpha_min = max(0.15 - 0.23 * (30 / next_ws) * (1 - (0.4 * np.log(130 / 110)**2)), self.alpha_min_limit)
        alpha_max = min(0.22 + 0.4 * (130 / 110) * (30 / next_ws), self.alpha_max_limit)
        alpha = np.random.uniform(0, 1) * (alpha_max - alpha_min) + alpha_min

        # Just to ensure we are keeping within limits...
        next_ws = np.clip(next_ws, 0, 30)
        ti = np.clip(ti, 0, 0.5)

        # next_ws = 8
        # ti = 0.1
        # alpha = 0.1

        info = {
            'wd': next_wd,
            'ws': next_ws,
            'ti': ti,
            'alpha': alpha,
            'vector': np.array([next_ws, next_wd, ti, alpha]),
            'other': {}
        }

        # Return the wind speed and direction
        return info

    def reset(self) -> dict:
        """
        Reset the wind sampler to a new initial state.
        It takes the equillibrium state to sample the 'previous' wind state from
        """

        self.prev_wd = np.random.multinomial(1, self.p_equillibrium).argmax()
        self.prev_ws = int(max(norm.rvs(loc=7, scale=2), 2))

        return self.step()

    def _determine_equillibrium(self) -> np.ndarray:
        """
        Determines the equillibrium state for the Markov Chain, given the transition matrix
        """

        n_states = self.p_transition_wd.shape[0]
        A = np.append(self.p_transition_wd[:, :, 0].T - np.eye(n_states), np.ones(n_states).reshape(1, -1), axis=0)
        pinv = np.linalg.pinv(A)
        return pinv.T[-1]

################################################################################################################
# TAKEN FROM: https://github.com/AlgTUDelft/wind-farm-env/blob/main/wind_farm_gym/wind_process/mvou_process.py #
# Gregory Neustroev, Sytze P.E. Andringa, Remco A. Verzijlbergh, Mathijs M. de Weerdt, 2022.                   #
################################################################################################################
class MultivariateOrnsteinUhlenbeck(WindSampler):
    # This is a data provider based on a Multi-Variate Ornstein-Uhlenbeck process. It is a continuous-time stochastic
    # process described by a differential equation
    # dX = theta (mu - X) dt + sigma dW,
    # where X is a vector of variables, theta is a drift matrix, mu is a mean vector, and sigma is a diffusion matrix.
    # There interpretation is as follows:
    # Names:     names of the variables in the same order they are used in vectors/matrices
    # Logs:      whether a logarithmic transformation has been taken or not. The transformation is usually required for
    #            non-negative variables, such as wind speed, to convert their co-domain from (0, inf) to (-inf, inf).
    #            in this example, the vector X = [X_1, X_2, X_3] represents logarithm of the turbulence intensity and
    #            wind speed M, X_1 = ln(TI), X_2 = ln(M), and wind direction phi, X_3 = phi.
    # Mean:      long-term mean values of the variables. E.g., wind speed M is exp(2.2769937) (i.e., approximately 9.75),
    #            and direction is 0 degrees from the mean wind direction (270).
    # Drift:     A drift matrix shows how fast the variables revert to the mean values after randomly drifting away.
    #            A zero matrix means no drift, and the variables are changing according to a brownian motion process.
    # Diffusion: Determines the scale of a random noise process. In this case the random noise is a 3-dimensional
    #            Brownian motion [W_1, W_2, W_3], where W_3 drives the randomness in the wind direction.
    #            The diffusion matrix governs the scale and dependencies on these two processes, so the wind direction
    #            (third line of the matrix) depends on W_3 only, but wind speed is influenced by both W_2 and W_3,
    #            i.e., random fluctuations in wind speed depend on random fluctuations in wind direction as well.
    # Mean Wind Direction: wind direction can be rotated arbitrarily. It is easier to simulate the wind with the mean
    #            direction of 0.0, and then rotate it by a given angle in degrees.

    DEFAULT_PROPERTIES = {
        'names': ['ti', 'ws', 'wd'],
        'logs': [True, True, False],
        'mean': [-2.1552094, 2.2769937, 0.0],
        'drift': [[0.0024904,      5.4994818e-04, -2.3334057e-06],
                [-2.1413137e-05, 4.7972649e-05,  5.2700795e-07],
                [3.0910895e-03, -3.57165e-03,    0.01]],
        'diffusion': [[0.0125682, -0.0002111, -0.0004371],
                    [0.0,        0.0021632,  0.0002508],
                    [0.0,        0.0,        0.1559985]],
        'mean_wind_direction': 270.0
    }

    def __init__(self, time_delta=1, properties=None, seed=None):
        """
        Initializes a MVOU process.

        :param time_delta: time interval between two consecutive time steps, in seconds
        :param properties: a dictionary of properties of the process, includes:
            `names`: a list of atmospheric conditions, e.g., ['wind_speed', 'wind_direction'];
            `logs`: a list of boolean values of the same length, showing whether logarithmic transformation is needed
                for a particular atmospheric measurement;
            `mean`: a list of mean values to which (log)variables tend to revert to; for wind_direction usually 0.0
            `drift`: the drift matrix
            `diffusion`: the diffusion matrix
            `mean_wind_direction`: wind direction is additionally rotated by this angle after the simulation; this makes
                it possible to turn the wind without needing to re-estimate the drift and diffusion
        :param seed: random seed
        """
        super().__init__()

        warn('MultivariateOrnsteinUhlenbeck is untested and may not work as expected. Use with caution.')

        self._seed = seed
        if properties is None:
            properties = self.DEFAULT_PROPERTIES
        self._dt = time_delta
        self._names = properties.get('names', [])
        self._theta = np.array(properties.get('drift', []))
        self._sigma = np.array(properties.get('diffusion', []))
        self._mu = np.array(properties.get('mean', []))
        assert len(self._theta.shape) == 2 and len(self._sigma.shape) == 2,\
            'Need square drift and diffusion matrices'
        assert self._theta.shape[0] == self._theta.shape[1] == self._sigma.shape[0] == self._sigma.shape[1],\
            'Matrices have incompatible dimensions'
        self._n = self._theta.shape[0]
        self._logs = properties.get('logs', [False for _ in range(self._n)])
        if self._mu is None:
            self._mu = np.zeros(self._n)
        else:
            assert len(self._mu.shape) == 1 and self._mu.size == self._n, f'Mean must be a vector of length {self._n}'
        self._mean_wind_direction = properties.get('mean_wind_direction', 270.0)
        self._np_random, self._x = None, None
        self.reset()

        # Based on
        # Meucci, A. (2009). Review of statistical arbitrage, cointegration, and multivariate Ornstein-Uhlenbeck.
        # http://symmys.com/node/132. Working Paper
        # and https://doi.org/10.1186/s13662-019-2214-1
        i_n = np.eye(self._n)
        self._exp_theta = expm(self._theta * (-self._dt))
        self._eps_mean = (i_n - self._exp_theta) @ self._mu
        kron_sum = np.kron(self._theta, i_n) + np.kron(i_n, self._theta)
        sigma_square = self._sigma @ self._sigma.transpose()
        self._eps_cov = pinv(kron_sum) @ (np.eye(self._n*self._n) - expm(kron_sum * (-self._dt)))
        self._eps_cov = self._eps_cov @ sigma_square.flatten('F')
        self._eps_cov = self._eps_cov.reshape((self._n, self._n), order='F')

    def step(self):
        eps = self._np_random.multivariate_normal(mean=self._eps_mean, cov=self._eps_cov)
        self._x = self._exp_theta @ self._x + eps
        return self._get_vars_dictionary()

    def reset(self):
        self._np_random, _ = seeding.np_random(self._seed)
        self._x = self._mu

    def _get_vars_dictionary(self):
        x_dict = {self._names[i]: (math.exp(self._x[i]) if self._logs[i] else self._x[i]) for i in range(len(self._x))}
        x_dict['ti'] = min(x_dict['ti'], 0.5)
        if 'wd' in self._names:
            x_dict['wd'] = (x_dict['wd'] + self._mean_wind_direction) % 360
        x_dict['vector'] = np.array([x_dict['wd'], x_dict['ws'], x_dict['ti']])
        x_dict['other'] = {}
        return x_dict

    def save(self, file, timesteps=10000):
        keys = self._names
        with open(file, 'w') as output_file:
            dict_writer = DictWriter(output_file, keys)
            dict_writer.writeheader()
            self.reset()
            for i in range(timesteps):
                dict_writer.writerow(self.step())

    @staticmethod
    def switch_to_csv(data_file, time_steps, time_delta, properties, seed):
        """
        Saves the MVOU process into a `.csv` file, and returns a CSVProcess that reads data from that file.
        Do this when the same process needs to be used multiple times to ensure that the same data is used.
        :param data_file: file to save the generated data into
        :param time_steps: number of time steps to save
        :param time_delta: time increment between two consecutive time steps, in seconds
        :param properties: a property dictionary for the MVOU process
        :param seed: random seed
        :return: a `CSVProcess` that reads data from the file
        """
        if not os.path.exists(data_file):
            wind_process = MultivariateOrnsteinUhlenbeck(time_delta=time_delta, properties=properties, seed=seed)
            wind_process.save(data_file, time_steps)
        return CSVSampler(data_file)