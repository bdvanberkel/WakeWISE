import numpy as np

from WakeWISE.DegradationModels import FatigueModel
from WakeWISE.WindTurbine.Turbine import Turbine


class IEA37Turbine(Turbine):
    """
    A wind turbine model based on the IEA 3.4 MW reference turbine
    Note that this model is very general, and might as well be used for any turbine
    However, the GNN surrogate model is trained on the IEA 3.4 MW reference turbine
    """

    def __init__(self, idx: int, deg_model: FatigueModel, deg_params: dict, **kwargs):
        """
        Initialises the wind turbine model

        Args:
            idx (int): the index of the wind turbine
            deg_model (FatigueModel): the fatigue model to use
            config (dict): the configuration for the wind turbine
        """

        super().__init__(idx, deg_params)

        # Set the degradation model for each component
        self.degradation_models = [deg_model(self.deg_params['components'][c], **{**self.deg_params['extra'], **kwargs}) for c in self.deg_params['components']]

        # Reset the turbine
        self.reset()

    @property
    def turbine_type(self):
        return 'IEA 3.4 MW'
    
    @property
    def cutin(self):
        return 4.0
    
    @property
    def cutout(self):
        return 25.0

    def update(self, DELs: dict, n: int) -> None:
        """
        Updates the damage for the wind turbine based on the given damage equivalent loads
        This is done for each component, with the given number of cycles

        Args:
            DELs (dict): the damage equivalent loads for each component
            n (int): the number of cycles
        """

        # If the turbine is not active, do not update the damage // DECIDED TO CHANGE: ALWAYS UPDATE
        # if not self.active:
        #     return

        # Check if the DELs and degradation components match
        assert DELs.shape[1] == len(self.degradation_models), 'Error: DELs and degradation components do not match'

        # DELs is of shape (1, n_components) or (2, n_components), depending on whether baseline detrending is used

        # Update the damage for each component
        for idx, component in enumerate(self.degradation_models):
            component.update_damage(DELs[:, idx, None], n)

    def reset(self) -> None:
        """
        Resets the status of the turbine, and resets the degradation models
        """

        self.active = True

        for component in self.degradation_models:
            component.reset_model()

    def get_D(self):
        """
        Gets the damage for each component
        """

        return {component['component']: component.D[0, 0] for component in self.degradation_models}
    
    def get_Ds(self):

        return np.array([component.D[0, 0] for component in self.degradation_models])
    
    def get_baseline_Ds(self):

        return np.array([component.D[1, 0] for component in self.degradation_models])
    
    def get_all_delta_Ds(self):

        return np.concatenate([component.delta_D for component in self.degradation_models], axis=1)
    
    def get_replacement_costs(self):
        """
        Gets the replacement costs for each component
        """

        return np.array([component.maintenance['replace']['cost'] for component in self.degradation_models])

    def __str__(self) -> str:
        """
        Returns a string representation of the wind turbine
        """
        return f'Turbine {self.idx} (Active: {str(self.active)}) with {len(self.degradation_models)} components:\n' + '\n'.join(['    '*2 + str(component) for component in self.degradation_models])
    
    def undo(self) -> None:
        """
        Undoes the last update to the damage
        """

        for component in self.degradation_models:
            component.undo()

