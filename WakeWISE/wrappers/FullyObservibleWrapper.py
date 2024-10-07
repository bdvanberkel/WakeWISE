import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.utils import RecordConstructorArgs


class FullyObservibleWrapper(ObservationWrapper, RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        """
        Make the environment fully observable, e.g. add all interesting state variables to the observation
        """

        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        self._define_spaces()

    def observation(self, obs):

        # Base environment shortcut
        benv = self.env.unwrapped

        # Get all damage states in a large tensor
        D = np.array([turbine.get_Ds() for turbine in benv.turbines.values()])

        # Based on whether we have a single observation or multiple, stack them or not
        if self.env.unwrapped.collective_observation:
            obs = np.hstack((benv.wind_condition['ws'], benv.wind_condition['wd'], benv.wind_condition['ti'], benv.wind_condition['alpha'], benv.rate, benv.windfarm_time.get_lifetime_fraction(), D.flatten())).astype(np.float32)
        else:
            obs = np.hstack(([[benv.wind_condition['ws'], benv.wind_condition['wd'], benv.wind_condition['ti'], benv.wind_condition['alpha'], benv.rate, benv.windfarm_time.get_lifetime_fraction()]] * D.shape[0], D)).astype(np.float32)

        return obs
    
    def _define_spaces(self):
        """
        Overwrite the base environments space constructor with the new ones
        """

        # Base environment shortcut
        benv = self.env.unwrapped

        # Shortcuts for parameters
        n_c = self.env.unwrapped.n_components
        n_t = self.env.unwrapped.n_turbines
        s_f = self.env.unwrapped.deg_params['extra'].get('damage_safety_factor', 1.00)

        # Construct observation spaces
        if self.env.unwrapped.collective_observation:
            lower_bounds = np.array([0, 0, 0, benv.wind_model.alpha_min_limit, benv.rate_model.min, 0] + [0] * n_c * n_t, dtype=np.float32)
            upper_bounds = np.array([30, 360, 0.5, benv.wind_model.alpha_max_limit, benv.rate_model.max, 1] + [1 / s_f] * n_c * n_t, dtype=np.float32)
            self.observation_space = Box(low=lower_bounds, high=upper_bounds, shape=(6 + n_c * n_t,), dtype=np.float32)
        else:
            lower_bounds = np.vstack([[0, 0, 0, benv.wind_model.alpha_min_limit, benv.rate_model.min, 0] + [0] * n_c] * n_t, dtype=np.float32)
            upper_bounds = np.vstack([[30, 360, 0.5, benv.wind_model.alpha_max_limit, benv.rate_model.max, 1] + [1 / s_f] * n_c] * n_t, dtype=np.float32)
            self.observation_space = Box(low=lower_bounds, high=upper_bounds, shape=(n_t, 6 + n_c), dtype=np.float32)

    def __str__(self) -> str:
        return f'FullyObservibleWrapper({self.env.unwrapped.__module__}) at {str(self.env.unwrapped.seasonal_time)} with {len(self.env.unwrapped.turbines)} turbines after {self.env.unwrapped.maintenance_model.interventions} maintenance interventions:\n' + '\n'.join(['    ' + str(self.env.unwrapped.turbines[idx]) for idx in self.env.unwrapped.turbines])