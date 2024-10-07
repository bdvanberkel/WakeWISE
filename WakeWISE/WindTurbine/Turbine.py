from abc import ABC, abstractmethod


class Turbine(ABC):

    def __init__(self, idx: int, deg_params: dict):

        # Set the id
        self.idx = idx

        # Set deg params
        self.deg_params = deg_params

    @property
    @abstractmethod
    def turbine_type(self) -> str:
        """
        The type of the turbine
        """

    @property
    @abstractmethod
    def cutin(self) -> float:
        """
        The cut-in wind speed for the turbine
        """

    @property
    @abstractmethod
    def cutout(self) -> float:
        """
        The cut-out wind speed for the turbine
        """

    @abstractmethod
    def update(self, DELs: dict, n: int) -> None:
        """
        Updates the damage for the wind turbine based on the given damage equivalent loads
        This is done for each component, with the given number of cycles

        Args:
            DELs (dict): the damage equivalent loads for each component
            n (int): the number of cycles
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the status of the turbine, and resets the degradation models
        """