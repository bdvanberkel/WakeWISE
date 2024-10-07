import numpy as np


class ParameterScheduler:

    def __init__(self, values: list, times: list, g, index) -> None:
        """
        Initializes the parameter scheduler

        Args:
            values (list): the values of the parameter
            times (list): the times at which the values change
            index (int | float): the index to use for interpolation
        """

        self.values = values
        self.times = times
        self.g = g
        self.index = index

        assert len(values) == len(times), "The number of values must be equal to the number of times"
        
        # Ensure 'index' exists as a property of class g
        assert hasattr(g, index), f"{index} is not a property of class g"

    def __float__(self) -> float:
        """
        Gets the value of the parameter, according to the value of g[index]

        Returns:
            float: the value of the parameter
        """

        return float(np.interp(getattr(self.g, self.index), self.times, self.values))