import gymnasium as gym


class DoNothingWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        """Template for a wrapper that does nothing to the environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

    def step(self, a):

        # Jep, boring.
        obs, reward, terminated, truncated, info = self.env.step(a)
        return obs, reward, terminated, truncated, info
