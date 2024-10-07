import numpy as np

from WakeWISE.MaintenanceModels.MaintenanceModel import MaintenanceModel


class SparseDoNothing(MaintenanceModel):
    """
    A maintenance model that does nothing

    This model is used as a template for the maintenance models
    """

    def __init__(self, g, **kwargs) -> None:
        
        super().__init__(g, **kwargs)

    def run(self, turbines: list, **conditions) -> float:

        if self.maintaining is None:
            # Infer the number of dimensions, turbines and components
            self.n_t = len(turbines)
            self.n_c = len(turbines[0].degradation_models)
            self.n_d = turbines[0].degradation_models[0].D.shape[0]
            self.maintaining = np.zeros((self.n_d, self.n_t, self.n_c))

        costs = np.zeros((self.n_t, 1))

        for idx in turbines.keys():
            turbines[idx].active = np.sum(self.maintaining[:, idx, :], axis=1)[:, None] < 1

        return costs

    def reset(self) -> None:
        pass

    def __str__(self) -> str:
        return f'{self.__class__.__name__} with no turbines under maintenance'

class SparseRunToFailure(MaintenanceModel):
    """
    A maintenance model that runs the turbines until they fail
    This model uses sparse rewards, meaning incurrence of costs only when a component fails
    """

    def __init__(self, g, **kwargs) -> None:

        super().__init__(g, **kwargs)

    def run(self, turbines: list, undo: bool = False, **conditions) -> float:
        """
        Run the maintenance model for the given turbines

        Args:
            turbines (list): the list of turbines
            undo (bool): whether to undo the maintenance actions (for gradient descent)

        Returns:
            float: the costs for the maintenance
        """

        if self.maintaining is None:
            # Infer the number of dimesions, turbines and components
            self.n_t = len(turbines)
            self.n_c = len(turbines[0].degradation_models)
            self.n_d = turbines[0].degradation_models[0].D.shape[0]
            self.maintaining = np.zeros((self.n_d, self.n_t, self.n_c))

        if not undo:
            
            # Reset components that have been repaired
            # Fetch the indices; i-dimension is for baseline removal, j-dimension is turbine, k-dimension is component
            for i, j, k in np.argwhere(self.maintaining == 1):

                # Set it under maintenance; D and delta_D were already reset when it was put under maintenance
                turbines[j].degradation_models[k].active[i] = True

            # Decrement the maintenance counter; ensure it does not drop below 0 (note: np.clip is not used to avoid unnecessary computation)
            self.maintaining = np.maximum(self.maintaining - 1, 0)

        # Pre-allocate the costs array
        costs = np.zeros((self.n_d, self.n_t, 1))

        for idx in turbines.keys():

            # This creates an (n_bootstraps, n_components) array with True if any component has reached the failure limit
            failures = np.concatenate([component.D for component in turbines[idx].degradation_models], axis=1) >= self.failure_limit

            # If any component, in baseline or 'true' dimensions, has reached the failure limit. Using .sum is faster than .any!
            if np.sum(failures):

                costs_i = np.zeros((self.n_d, 1))

                # Fetch the indices; i-dimension is true/baseline, j-dimension is component
                for i, j in np.argwhere(failures):

                    if not undo:

                        # Set it under maintenance for the correct duration
                        self.maintaining[i, idx, j] = turbines[idx].degradation_models[j].maintenance['replace']['duration']

                        # Reset D and delta_D, but keep under maintenance so it doesn't accumulate damage and costs
                        turbines[idx].degradation_models[j].reset_model(dim=i, maintenance=True)

                        # Increment the interventions
                        self.interventions += 1

                    # Increment the costs
                    costs_i[i] += turbines[idx].degradation_models[j].maintenance['replace']['cost']

                # Cost model follows c(D(t), delta_D(t)) = δ(D - 1) * cost
                # Dirac-Delta at D=1
                costs[:self.n_d, idx] = costs_i

            # If any components still has remaining maintenance, the turbine is not active
            # This essentially collapses the maintenance array to a (1,1) or (2,1) array with the sum over components, for a given turbine
            # If any component for that turbine had remaining maintenance time, this will evaluate to a non-zero value
            # Using .sum and a comparison is faster than .any!
            turbines[idx].active = np.sum(self.maintaining[:, idx, :], axis=1)[:, None] < 1

        # Two possibilities: without baseline removal, we take the value as-is; with baseline removal, we take the difference
        return costs.squeeze(axis=0) if self.n_d == 1 else np.diff(-costs, axis=0).squeeze(axis=0)
    
    def __str__(self) -> str:

        return f'{self.__class__.__name__} with {np.sum(self.maintaining > 0)} turbines under maintenance:\n' + '\n'.join([f'    Bootstrap {i}, Turbine {j}, component {k} under maintenance for {self.maintaining[i, j, k]} timesteps' for i, j, k in np.argwhere(self.maintaining > 0)])

    def reset(self) -> None:

        self.maintaining = None