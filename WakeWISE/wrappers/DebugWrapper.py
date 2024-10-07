from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from WakeWISE.utils.logger import log


class DebugWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        """
        Prints some episode-wise information to the console.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        # Store episode-wise information
        self.angles = []
        self.reward_total = 0

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        # Add yaw angles to storage
        self.angles += list(action)

        # Run step as normal
        obs, rew, ter, tru, inf = self.env.step(action)

        # Add reward to cumulative total
        self.reward_total += np.sum(rew)

        # Return everything as normal
        return obs, rew, ter, tru, inf
    
    def reset(self, seed: int = None, options: dict = None) -> Any:

        # If we have anything to debug log about, print it to terminal
        if len(self.angles) > 0:
            log(f"Yaw angle mean {np.mean(self.angles):.5f}, std {np.std(self.angles):.5f}, min {np.min(self.angles):.5f}, max {np.max(self.angles):.5f} with reward sum of {self.reward_total:.5f}", origin = "DebugWrapper.py")
        
        # Reset our storage
        self.angles = []
        self.reward_total = 0
        
        # Reset environment as usual
        return super().reset(seed = seed, options = options)