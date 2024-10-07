import numpy as np

from WakeWISE.DegradationModels.Stochastic import FatigueModel


class DeterministicFatigueModel(FatigueModel):
    """
    Deterministic fatigue model
    Models the fatigue using a deterministic approach
    The fatigue model is defined by the following parameters:
    - DEL_U: The ultimate load at which the component fails
    - m: The slope of the S-N curve
    The DEL_U and m are modelled as deterministic variables

    Upon initialization, the model will take the DEL_U and m from the given parameters
    These will be kept constant, even if the model is reset
    """

    def __init__(self, params, **kwargs):
        """
        Initializes the deterministic fatigue model for the given component

        Args:
            params (dict): the parameters for the fatigue model
        """

        assert {'del_u', 'm'}.issubset(params), f"Missing parameters for {self.__class__.__name__} in component {params['type']}"

        super().__init__(params['type'], **kwargs)

        # Set the parameters
        self.DEL_u = float(params['del_u'])
        self.m = float(params['m'])
        self.maintenance = params['maintenance']
        self.reset_model()

    def update_damage(self, DEL : np.ndarray, n : int):
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
        fatigue_limit = 1 / ((DEL / self.DEL_u) ** self.m)

        self.delta_D = n / fatigue_limit * self.active

        # Update the damage, based on the fraction of the fatigue limit accumulated in this timestep
        self.D = np.minimum(self.D + self.delta_D, self.failure_limit)
            
    def reset_model(self, dim=None, maintenance=False):
        """
        Resets the fatigue model

        Args:
            dim (int): the dimension to reset (e.g. baseline or not), if None, the entire model will be reset
            maintenance (bool): whether the component is put under maintenance
        """

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
        return f'{self.__class__.__name__:<25} for component {self.component:<15} (D={self.D[0, 0]:.8f}) with DEL_U = {round(self.DEL_u, 2):<10} and m = {round(self.m, 2):<5}'