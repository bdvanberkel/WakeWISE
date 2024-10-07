from typing import Callable

import gymnasium as gym
import numpy as np

from WakeWISE.utils.logger import warn


class ScaleRewardWrapper(gym.RewardWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, scaling_factor: float = None, f: Callable[[float], float] = None):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        gym.utils.RecordConstructorArgs.__init__(self, f=f)
        gym.RewardWrapper.__init__(self, env)

        if scaling_factor is not None:
            self.f = lambda r: r / scaling_factor
        elif f is not None:
            assert callable(f)
            self.f = f
        else:
            warn(f"Either scaling_factor or f must be provided; defaulting to 24000 * n_turbines = {self.env.unwrapped.n_turbines * 24000}", origin = 'ScaleRewardsWrapper.py')
            self.f = lambda r: r / (self.env.unwrapped.n_turbines * 24000)

    def reward(self, reward):
        """Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        """
        return self.f(reward)
