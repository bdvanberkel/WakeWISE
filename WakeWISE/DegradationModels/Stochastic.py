from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from WakeWISE.utils.logger import warn


class FatigueModel(ABC):
    """
    Base class for a fatigue model
    """

    def __init__(self, component: str, **kwargs) -> None:
        """
        Initializes the fatigue model for the given component

        Args:
            component (str): the component for which the fatigue model is being initialized
        """

        # Set the component name
        self.component = component

        # Set the active status
        self.active = None

        # Set the safety factor
        self.sf = kwargs.get('damage_safety_factor', 1.00)
        self.failure_limit = 1 / self.sf

        # Bottom and top of the random initialisation range
        self.random_D_initialisation = kwargs.get('random_D_initialisation', False)
        self.random_D_initialisation_range = kwargs.get('random_D_initialisation_range', [0, self.failure_limit])

        if self.random_D_initialisation_range[0] > self.random_D_initialisation_range[1]:
            warn(f"'low' of random_initialisation_range is greater than 'high'. Setting 'low' to 0 and 'high' to {self.failure_limit}")
            self.random_D_initialisation_range = [0, self.failure_limit]
        if self.random_D_initialisation_range[1] > self.failure_limit:
            warn(f"'high' of random_initialisation_range is greater than the failure limit. Setting 'high' to {self.failure_limit}")
            self.random_D_initialisation_range[1] = self.failure_limit
        if self.random_D_initialisation_range[0] < 0:
            warn("'low' of random_initialisation_range is less than 0. Setting 'low' to 0")
            self.random_D_initialisation_range[0] = 0

        # Set the damage to 0
        self.D = np.random.uniform(self.random_D_initialisation_range[0], self.random_D_initialisation_range[1], size=(1, 1)) if self.random_D_initialisation else np.array([[0.00]])

        # Set the change in D to 0
        self.delta_D = np.zeros_like(self.D)

    @abstractmethod
    def update_damage(self, DEL: np.ndarray, n: int) -> None:
        """
        Updates the damage for the given component with the given DEL and number of cycles
        Must relate the DEL and the number of cycles to the damage

        Args:
            DEL (int): the damage equivalent load
            n (int): the number of cycles
        """

    def reset_model(self) -> None:
        """
        Resets the fatigue model to its initial state
        """

        self.D = np.zeros_like(self.D)

    def __str__(self) -> str:
        """
        Returns a string representation of the fatigue model
        """
        return f'{self.__class__.__name__:<25} for component {self.component:<15} with D = {self.D[0, 0]:.8f}'
    
    def undo(self) -> None:
        """
        Undoes the last update to the damage
        """
        self.D -= self.delta_D


class StochasticFatigueModel(FatigueModel):
    """
    Stochastic fatigue model
    Models the fatigue using a somewhat stochastic approach
    The fatigue model is defined by the following parameters:
    - DEL_U: The ultimate load at which the component fails
    - m: The slope of the S-N curve
    The DEL_U and m are modelled as random variables, with a mean and standard deviation

    Upon initialization, the model will sample the DEL_U and m from the given distributions
    These will be kept constant for the lifetime of the component, till the model is reset
    """

    def __init__(self, params, **kwargs):
        """
        Initializes the stochastic fatigue model for the given component

        Args:
            params (dict): the parameters for the fatigue model
        """

        assert {'del_u_mean', 'del_u_std', 'm_mean', 'm_std'}.issubset(params), f"Missing parameters for {self.__class__.__name__} in component {params['type']}"

        super().__init__(params['type'])

        # Set the parameters
        self.DEL_U_mean = params['del_u_mean']
        self.DEL_U_std = params['del_u_std']
        self.m_mean = params['m_mean']
        self.m_std = params['m_std']

        # Sample the DEL_U and m
        self.reset_model()

    def update_damage(self, DEL: np.ndarray, n: int) -> None:
        """
        Updates the damage for the given component with the given DEL and number of cycles

        Args:
            DEL (int): the damage equivalent load
            n (int): the number of cycles
        """

        # DEL is of shape (1, 1) or (2, 1), depending on whether baseline removal is used

        if self.active is None:
            self.active = np.full((DEL.shape[0], 1), True)

        # Calculate the fatigue limit
        fatigue_limit = 1 / ((DEL / self.current_DEL_U) ** self.current_m)

        self.delta_D = n / fatigue_limit * self.active

        # Update the damage, based on the fraction of the fatigue limit accumulated in this timestep
        self.D = np.minimum(self.D + self.delta_D, self.failure_limit)

    def reset_model(self, dim=None, maintenance=False) -> None:
        """
        Resets the fatigue model with new DEL_U and m samples

        Args:
            dim (int): the dimension to reset (e.g. baseline or not), if None, the entire model will be reset
            maintenance (bool): whether the component is put under maintenance
        """

        # Sample the DEL_U and m    
        self.current_DEL_U = norm.rvs(loc=self.DEL_U_mean, scale=self.DEL_U_std)
        self.current_m = norm.rvs(loc=self.m_mean, scale=self.m_std)

        if dim is not None:
            assert dim < len(self.D), f"Dimension {dim} is out of bounds for component {self.component}"
            self.D[dim] = 0.00
            self.delta_D[dim] = 0.00
            self.active[dim] = not maintenance
        else:
            self.D = np.random.uniform(self.random_D_initialisation_range[0], self.random_D_initialisation_range[1], size=(1, 1)) if self.random_D_initialisation else np.zeros_like(self.D)
            self.delta_D = np.zeros_like(self.D)
            self.active = None

    def __str__(self) -> str:
        return f'{self.__class__.__name__:<25} for component {self.component:<15} (D={self.D[0, 0]:.8f}) with DEL_U = {round(self.current_DEL_U, 2):<10} and m = {round(self.current_m, 2):<5}'