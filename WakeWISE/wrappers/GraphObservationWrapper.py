import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.utils import RecordConstructorArgs

from WakeWISE.utils.spaces import CustomGraph


class GraphObservationWrapper(ObservationWrapper, RecordConstructorArgs):
    """
    Wrapper to embed the observation of the environment onto a graph.
    Note that this will only work if collective (single-vector) observations are turned OFF,
    since it expects a two-dimensional tensor to distribute observations over the graph as
    node features.
    """

    def __init__(self, env: gym.Env):

        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        if self.env.unwrapped.collective_observation:
            raise ValueError('GraphObservationWrapper only supports the individual observation version of the environment. Please set multi_agent_collective_observation=False in the environment constructor. (has to do with the way the observation is constructed)')

        self._define_spaces()

    def observation(self, obs):

        # Return a dict-like observation, we call a 'CustomGraph' space.
        return {
            'node_features': obs,
            'edge_features': self.env.unwrapped.model.graph_layout.edge_attr.detach().clone().cpu().numpy().astype(np.float32),
            'edge_links': self.env.unwrapped.model.graph_layout.edge_index.detach().clone().cpu().numpy().astype(np.int32)
        }

    def _define_spaces(self):

        # Since we re-use the edge connectivity and edge features of the GNN surrogate, obtain the lower and upper bounds from the surrogate
        edge_feat_lower_bounds = np.array(self.env.unwrapped.model.model.trainset_stats['edges']['min'].clone().detach().cpu().numpy(), dtype=np.float32)
        edge_feat_upper_bounds = np.array(self.env.unwrapped.model.model.trainset_stats['edges']['max'].clone().detach().cpu().numpy(), dtype=np.float32)

        # Construct our custom graph observation space.
        self.observation_space = CustomGraph(
            num_nodes = self.env.unwrapped.n_turbines,
            num_edges = self.env.unwrapped.model.num_edges,
            node_space = Box(low=self.observation_space.low[0, :], high=self.observation_space.high[0, :], shape=self.observation_space.low[0, :].shape, dtype=np.float32),
            edge_space = Box(low=edge_feat_lower_bounds, high=edge_feat_upper_bounds, shape=(self.env.unwrapped.model.edge_dim,), dtype=np.float32)
        )