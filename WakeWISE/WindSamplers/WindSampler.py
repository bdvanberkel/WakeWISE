from abc import ABC, abstractmethod

from WakeWISE.utils import TimeKeeper


class WindSampler(ABC):
    """
    Base class for a detailed wind sampler

    """

    @abstractmethod
    def step(self, time: TimeKeeper) -> dict:
        """
        Samples the wind speed and direction for the given time

        Args:
            time (TimeKeeper): the time for which to sample the wind

        Returns:
            dict: dict containing wind direction, speed and turbulence intensity. Also allows to pass other environmental conditions in 'other'
        """

    @abstractmethod
    def reset(self, time: TimeKeeper) -> dict:
        """
        Reset the wind sampler to its initial state
        """

    def __str__(self) -> str:
        """
        Returns a string representation of the wind sampler

        Returns:
            str: the string representation
        """
        return f'{self.__class__.__name__}'