from typing import Any

import gymnasium as gym


class RandomiseFarmWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        """
        Randomises the farm layout at the start of each episode.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
    
    def reset(self, seed: int = None, options: dict = None) -> Any:

        self.env.unwrapped.model.set_layout()
        return super().reset(seed = seed, options = options)