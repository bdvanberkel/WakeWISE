import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.utils import RecordConstructorArgs

from WakeWISE.utils.normalizer import Normalizer
from WakeWISE.utils.spaces import CustomGraph


class ObsNormalisationWrapper(ObservationWrapper, RecordConstructorArgs):
    """
    Wrapper to normalise the observation space. Different from Gymnasium's wrapper
    because we have a custom graph observation space which needs custom normalisation.
    """

    def __init__(self, env: gym.Env):

        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        self._define_spaces()

    def observation(self, obs):

        # Normalise the observation, but be careful to choose the right type of normalisation
        if isinstance(self.observation_space, Box):
            return self.observation_normaliser.normalize(obs)
        elif isinstance(self.observation_space, CustomGraph):
            obs['node_features'] = self.node_normaliser.normalize(obs['node_features'])
            obs['edge_features'] = self.edge_normaliser.normalize(obs['edge_features'])
            return obs
    
    def __str__(self) -> str:
        return f'ObsNormalisationWrapper({self.env.unwrapped.__module__}) at {str(self.env.unwrapped.seasonal_time)} with {len(self.env.unwrapped.turbines)} turbines after {self.env.unwrapped.maintenance_model.interventions} maintenance interventions:\n' + '\n'.join(['    ' + str(self.env.unwrapped.turbines[idx]) for idx in self.env.unwrapped.turbines])
    
    def _define_spaces(self):

        # Overwrite the spaces
        if isinstance(self.observation_space, Box):
            self.observation_normaliser = Normalizer(self.observation_space.low, self.observation_space.high)

            new_low = np.zeros(self.observation_space.shape)
            new_high = np.ones(self.observation_space.shape)
            self.observation_space = Box(low=new_low, high=new_high, shape=self.observation_space.shape, dtype=np.float32)

        # Same, but for the custom graph observation space
        elif isinstance(self.observation_space, CustomGraph):
            self.node_normaliser = Normalizer(self.observation_space.spaces['node_features'].low, self.observation_space.spaces['node_features'].high)
            self.edge_normaliser = Normalizer(self.observation_space.spaces['edge_features'].low, self.observation_space.spaces['edge_features'].high)

            self.observation_space = CustomGraph(
                num_nodes = self.observation_space.num_nodes,
                num_edges = self.observation_space.num_edges,
                node_space = Box(low=0, high=1, shape=self.observation_space.spaces['node_features'].shape[1:], dtype=np.float32),
                edge_space = Box(low=0, high=1, shape=self.observation_space.spaces['edge_features'].shape[1:], dtype=np.float32)
            )

        else:
            raise ValueError(f"Observation space '{type(self.observation_space)}' not supported for normalisation")

class ActionNormalisationWrapper(ActionWrapper, RecordConstructorArgs):
    """
    Wrapper to normalise the action space
    """

    def __init__(self, env: gym.Env):

        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        self._define_spaces()

    def action(self, action):

        return self.action_normaliser.denormalize(action)
    
    def __str__(self) -> str:
        return f'FullyObservibleWrapper({self.env.unwrapped.__module__}) at {str(self.env.unwrapped.seasonal_time)} with {len(self.env.unwrapped.turbines)} turbines after {self.env.unwrapped.maintenance_model.interventions} maintenance interventions:\n' + '\n'.join(['    ' + str(self.env.unwrapped.turbines[idx]) for idx in self.env.unwrapped.turbines])
    
    def _define_spaces(self):

        if isinstance(self.action_space, Box):
            self.action_normaliser = Normalizer(self.action_space.low, self.action_space.high)

            new_low = np.zeros(self.action_space.shape)
            new_high = np.ones(self.action_space.shape)
            self.action_space = Box(low=new_low, high=new_high, shape=self.action_space.shape, dtype=np.float32)