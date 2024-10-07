from abc import ABC, abstractmethod

import numpy as np

from typing import Any


class PropagationModel(ABC):

    def __init__(self):

        # Number of turbines in the wind farm, define in subclass
        self.n_turbines = None

    @abstractmethod
    def __call__(self, wind_condition: np.ndarray, turbine_states: np.ndarray) -> np.ndarray:
        """
        Chain function for the propagation model, calls the run function
        """

    @abstractmethod
    def run(self, wind_condition: np.ndarray, turbine_states: np.ndarray) -> np.ndarray:
        """
        Run the surrogate model to get the power output, local ws and ti and the DEL values for the wind farm layout
        
        Args:
            wind_condition (np.ndarray): the wind condition for the wind farm
            turbine_states (np.ndarray): the states of the turbines in the wind farm

        Returns:
            np.ndarray: the power output, local ws and ti and the DEL values for the wind farm layout
        """

    def get_positions(self) -> Any:
        """
        Get the positions of the turbines in the wind farm
        """