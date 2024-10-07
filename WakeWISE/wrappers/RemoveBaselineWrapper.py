import gymnasium as gym
import numpy as np

from WakeWISE.utils.BaselinePowerSurrogate import BaselinePowerSurrogate


class RemoveBaselineWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        """Removes a baseline reward from the reward returned by the environment.
        In this case, the expected 'optimal' power at zero-yaw and wake-free
        conditions is removed from the produced power at each turbine. In a farm
        with zero wakes, the reward after removal of the baseline would be (near) zero.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self.model = BaselinePowerSurrogate(self.env.unwrapped.turbines[0])

    def step(self, a):

        curr_wind = self.env.unwrapped.wind_condition
        curr_rate = self.env.unwrapped.rate

        obs, rew, ter, tru, inf = self.env.step(a)

        IDEAL_POWER = self.model(curr_wind)

        if self.env.unwrapped.collective_reward:
            rew -= IDEAL_POWER * self.env.unwrapped.timestep_seconds / 3_600_000 * curr_rate * self.env.unwrapped.n_turbines
        else:
            rew -= IDEAL_POWER * self.env.unwrapped.timestep_seconds / 3_600_000 * curr_rate

        return obs, rew, ter, tru, inf
    

class RemoveBootstrappedBaselineWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, baseline_policy: np.ndarray = None):
        """Removes a baseline reward from the reward returned by the environment.
        In this case, we bootstrap a static baseline policy in all calculations
        We then remove the expected 'optimal' power at zero-yaw and wake-free
        conditions from the produced power at each turbine. Furthermore, we
        remove the cost of the baseline policy from the reward.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        if baseline_policy is None:
            baseline_policy = np.zeros((self.env.unwrapped.n_turbines,))

        assert baseline_policy.shape == (self.env.unwrapped.n_turbines,), "Baseline policy must have the same number of turbines as the environment"

        self.baseline_policy = baseline_policy

    def step(self, a):

        a = np.vstack((a, self.baseline_policy))

        obs, rew, ter, tru, inf = self.env.step(a)

        return obs, rew, ter, tru, inf

