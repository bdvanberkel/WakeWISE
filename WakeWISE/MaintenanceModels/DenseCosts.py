import numpy as np
from scipy.special import lambertw

from WakeWISE.MaintenanceModels.SparseCosts import MaintenanceModel
from WakeWISE.utils.scheduler import ParameterScheduler

from WakeWISE.utils.logger import warn


class DenseRunToFailure(MaintenanceModel):
    """
    A maintenance model that runs the turbines until they fail
    This model uses dense rewards, meaning incurrence of costs at each timestep
    based on an even distribution of costs over the entire damage domain
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
            # Infer the number of dimensions (if baseline removal), turbines and components
            self.n_t = len(turbines)
            self.n_c = len(turbines[0].degradation_models)
            self.n_d = turbines[0].degradation_models[0].D.shape[0]
            self.maintaining = np.zeros((self.n_b, self.n_t, self.n_c))

        if not undo:
            
            # Reset components that have been repaired
            # Fetch the indices; i-dimension is for baseline removal, j-dimension is turbine, k-dimension is component
            for i, j, k in np.argwhere(self.maintaining == 1):

                # Set it out of maintenance; D and delta_D were already reset when it was put under maintenance
                turbines[j].degradation_models[k].active[i] = True

            # Decrement the maintenance counter; ensure it does not drop below 0 (note: np.clip is not used to avoid unnecessary computation)
            self.maintaining = np.maximum(self.maintaining - 1, 0)

        # Pre-allocate the costs array
        costs = np.zeros((self.n_d, self.n_t, 1))

        for idx in turbines.keys():

            # Cost model follows c(D(t), delta_D(t)) = delta_D(t) / failure_limit * cost
            # Even distribution of costs over entire damage domain
            costs[:self.n_d, idx] = np.sum([component.delta_D / self.failure_limit * component.maintenance['replace']['cost'] for component in turbines[idx].degradation_models], axis=0)

            # If undo, skip the rest of the loop; used for gradient descent, where we might only want to calculate the costs
            if undo:
                continue

            # This creates an (n_dimensions, n_components) array with True if any component has reached the failure limit
            failures = np.concatenate([component.D for component in turbines[idx].degradation_models], axis=1) >= self.failure_limit

            # If any component, in baseline or 'true' dimensions, has reached the failure limit. Using .sum is faster than .any!
            if np.sum(failures):

                # Fetch the indices; i-dimension is real/baseline, j-dimension is component
                for i, j in np.argwhere(failures):

                    # Set it under maintenance for the correct duration
                    self.maintaining[i, idx, j] = turbines[idx].degradation_models[j].maintenance['replace']['duration']

                    # Reset D and delta_D, but keep under maintenance so it doesn't accumulate damage and costs
                    turbines[idx].degradation_models[j].reset_model(dim=i, maintenance=True)

                    # Increment the interventions
                    self.interventions += 1

            # If any components still has remaining maintenance, the turbine is not active
            # This essentially collapses the maintenance array to a (1,1) or (2,1) array with the sum over components, for a given turbine
            # If any component for that turbine had remaining maintenance time, this will evaluate to a non-zero value
            # Using .sum and a comparison is faster than .any!
            turbines[idx].active = np.sum(self.maintaining[:, idx, :], axis=1)[:, None] < 1

        # Two possibilities: without baseline removal, we take the value as-is; with baseline removal, we take the difference
        return costs.squeeze(axis=0) if self.n_b == 1 else np.diff(-costs, axis=0).squeeze(axis=0)
    
    def __str__(self) -> str:

        return f'{self.__class__.__name__} with {np.sum(self.maintaining > 0)} turbines under maintenance:\n' + '\n'.join([f'    Bootstrap {i}, Turbine {j}, component {k} under maintenance for {self.maintaining[i, j, k]} timesteps' for i, j, k in np.argwhere(self.maintaining > 0)])
    
    def reset(self) -> None:

        self.maintaining = None

class ExponentialDenseRunToFailureV1(MaintenanceModel):
    """
    A maintenance model that runs the turbines until they fail
    This model uses dense rewards, meaning incurrence of costs at each timestep
    However, compared to the DenseRunToFailure model, the costs are calculated using a non-linear function
    Important to note that the costs in the end still sum up to the same amount as in the DenseRunToFailure model
    """

    def __init__(self, g, **kwargs) -> None:

        super().__init__(g, **kwargs)

        # f(x) = e^cx - 1
        if kwargs.get('exponential_c_schedule', False):
            self.c_getter = ParameterScheduler(kwargs.get('exponential_c_schedule_values', [1.25]), kwargs.get('exponential_c_schedule_times', [0]), self.g, 'EPISODE')
        else:
            self.c_getter = kwargs.get('exponential_c', 1.25)

        # Legacy code; initially, the original integral value was calculated beforehand. This is no longer necessary,
        # as I simplified the calculation and merged the two formulas together, no longer requiring a two step calculation
        # self.original_integral = ((1 / self.c) * np.exp(self.c * self.failure_limit) - self.failure_limit) - (1 / self.c)

    def __determine_cost_factor(self, D: float, delta_D: float) -> float:

        # Legacy code: see comment in __init__ for explanation. Left in for reference as it is easier to understand
        # Must use D - delta_D as the left bound of integration, as deg models have already been updated with the new damage
        # delta_integral = ((1 / self.c) * np.exp(self.c * D) - D) - ((1 / self.c) * np.exp(self.c * (D - delta_D)) - (D - delta_D))
        # return delta_integral / self.original_integral

        return (np.exp(self.c * D) - self.c * D - np.exp(self.c * (D - delta_D)) + self.c * (D - delta_D)) / (np.exp(self.c * self.failure_limit) - 1 - self.c * self.failure_limit)

    def run(self, turbines: list, undo: bool = False, **conditions) -> float:
        """
        Run the maintenance model for the given turbines

        Args:
            turbines (list): the list of turbines
            undo (bool): whether to undo the maintenance actions (for gradient descent)

        Returns:
            float: the costs for the maintenance
        """

        self.c = float(self.c_getter)

        if self.maintaining is None:
            # Infer the number of dimensions, turbines and components
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

            # Cost model follows c(D(t), delta_D(t)) = nonlin(D(t), delta_D(t)) * cost
            # Nonlinear (uneven) distribution of costs over entire damage domain
            costs[:self.n_d, idx] = np.sum([self.__determine_cost_factor(component.D, component.delta_D) * component.maintenance['replace']['cost'] for component in turbines[idx].degradation_models], axis=0)

            # If undo, skip the rest of the loop; used for gradient descent, where we might only want to calculate the costs
            if undo:
                continue

            # This creates an (n_dimensions, n_components) array with True if any component has reached the failure limit
            failures = np.concatenate([component.D for component in turbines[idx].degradation_models], axis=1) >= self.failure_limit

            # If any component, in baseline or 'true' dimensions, has reached the failure limit. Using .sum is faster than .any!
            if np.sum(failures):

                # Fetch the indices; i-dimension is true/baseline, j-dimension is component
                for i, j in np.argwhere(failures):

                    # Set it under maintenance for the correct duration
                    self.maintaining[i, idx, j] = turbines[idx].degradation_models[j].maintenance['replace']['duration']

                    # Reset D and delta_D, but keep under maintenance so it doesn't accumulate damage and costs
                    turbines[idx].degradation_models[j].reset_model(dim=i, maintenance=True)

                    # Increment the interventions
                    self.interventions += 1

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

class ExponentialDenseRunToFailureV2(MaintenanceModel):
    """
    A maintenance model that runs the turbines until they fail
    This model uses dense rewards, meaning incurrence of costs at each timestep
    However, compared to the DenseRunToFailure model, the costs are calculated using a non-linear function
    Important to note that the costs in the end still sum up to the same amount as in the DenseRunToFailure model
    """

    def __init__(self, g, **kwargs) -> None:

        super().__init__(g, **kwargs)

        # f(x) = e^cx - p
        if kwargs.get('exponential_c_schedule', False):
            self.c_getter = ParameterScheduler(kwargs.get('exponential_c_schedule_values', [1.25]), kwargs.get('exponential_c_schedule_times', [0]), self.g, 'EPISODE')
        else:
            self.c_getter = kwargs.get('exponential_c', 1.25)

    def __determine_p(self, c: float) -> float:
        if c == 0:
            return 0
        elif c <= 1.25:
            return (np.exp(c) - c - 1) / c
        else:
            return np.exp(lambertw((c - np.exp(c)) / np.exp(c + 1), k=-1).real + c + 1)

    def __determine_cost_factor(self, D: float, delta_D: float) -> float:
        if self.c == 0:
            return delta_D
        else:
            a = D
            b = D + delta_D
            intersect = np.log(self.p) / self.c
            if np.sum(a < intersect):
                return np.zeros_like(a)
            else:
                surf = ((1 / self.c) * np.exp(self.c * b) - self.p * b) - ((1 / self.c) * np.exp(self.c * a) - self.p * a)
                return surf

    def run(self, turbines: list, undo: bool = False, **conditions) -> float:
        """
        Run the maintenance model for the given turbines

        Args:
            turbines (list): the list of turbines
            undo (bool): whether to undo the maintenance actions (for gradient descent)

        Returns:
            float: the costs for the maintenance
        """

        self.c = float(self.c_getter)
        self.p = self.__determine_p(self.c)

        if self.maintaining is None:
            # Infer the number of bootstraps, turbines and components
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

            # Cost model follows c(D(t), delta_D(t)) = nonlin(D(t), delta_D(t)) * cost
            # Nonlinear (uneven) distribution of costs over entire damage domain
            costs[:self.n_d, idx] = np.sum([self.__determine_cost_factor(component.D, component.delta_D) * component.maintenance['replace']['cost'] for component in turbines[idx].degradation_models], axis=0)

            # If undo, skip the rest of the loop; used for gradient descent, where we might only want to calculate the costs
            if undo:
                continue

            # This creates an (n_dimensions, n_components) array with True if any component has reached the failure limit
            failures = np.concatenate([component.D for component in turbines[idx].degradation_models], axis=1) >= self.failure_limit

            # If any component, in baseline or 'true' dimensions, has reached the failure limit. Using .sum is faster than .any!
            if np.sum(failures):

                # Fetch the indices; i-dimension is true/baseline, j-dimension is component
                for i, j in np.argwhere(failures):

                    # Set it under maintenance for the correct duration
                    self.maintaining[i, idx, j] = turbines[idx].degradation_models[j].maintenance['replace']['duration']

                    # Reset D and delta_D, but keep under maintenance so it doesn't accumulate damage and costs
                    turbines[idx].degradation_models[j].reset_model(dim=i, maintenance=True)

                    # Increment the interventions
                    self.interventions += 1

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

class ExtrapolatedDenseRunToFailure(MaintenanceModel):
    """
    A maintenance model that runs the turbines until they fail
    This model uses dense rewards, meaning incurrence of costs at each timestep
    However, it extrapolates the costs till the end of the horizon based on a running mean
    It mimics sparse costs by examining whether the extrapolated cost results in a failure
    and consequently spreads that cost out over the horizon.
    """

    def __init__(self, g, **kwargs) -> None:

        super().__init__(g, **kwargs)
        self.running_delta_D_mean = 0
        self.iter = 0
        
        warn('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+', origin='DenseCosts.py')
        warn('ExtrapolatedDenseRunToFailure is still in development and should be used with caution', origin='DenseCosts.py')
        warn('For example: costs are extrapolated, but components are still reset upon failure', origin='DenseCosts.py')
        warn('resulting in biased extrapolations. This was more of an experimental module', origin='DenseCosts.py')
        warn('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+', origin='DenseCosts.py')

    def run(self, turbines: list, undo: bool = False, **conditions) -> int:
        """
        Run the maintenance model for the given turbines

        Args:
            turbines (list): the list of turbines
            undo (bool): whether to undo the maintenance actions (for gradient descent)

        Returns:
            float: the costs for the maintenance
        """

        if self.maintaining is None:
            # Infer the number of dimensions, turbines and components
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
        # costs = np.zeros((self.n_b, self.n_t, 1))

        # Update counter to calculate running mean
        self.iter += 1

        # Update the running mean of delta_D
        delta_Ds = np.stack([turbine.get_all_delta_Ds() for turbine in turbines.values()], axis=1)
        self.running_delta_D_mean += (delta_Ds - self.running_delta_D_mean) / self.iter

        # Get the current D and extrapolate it to the end of the horizon
        current_D = np.array([turbine.get_Ds() for turbine in turbines.values()])
        delta_lifetime_fraction = 1 / (20 * 365 * 24 * 6)
        extrapolated_D = current_D + self.running_delta_D_mean * (1 - self.g.FRACTION) / delta_lifetime_fraction

        # Get the replacement costs (parameters)
        replacement_costs = np.array([turbine.get_replacement_costs() for turbine in turbines.values()])

        # Extrapolate the costs to the end of the horizon
        costs = np.floor(extrapolated_D) * replacement_costs * delta_lifetime_fraction
        costs = np.sum(costs, axis=2)[:, :, None]
        assert costs.shape == (self.n_d, self.n_t, 1)

        for idx in turbines.keys():

            # If undo, skip the rest of the loop; used for gradient descent, where we might only want to calculate the costs
            if undo:
                continue

            # This creates an (n_bootstraps, n_components) array with True if any component has reached the failure limit
            failures = np.concatenate([component.D for component in turbines[idx].degradation_models], axis=1) >= self.failure_limit

            # If any component, in baseline or real dimensions, has reached the failure limit. Using .sum is faster than .any!
            if np.sum(failures):

                # Fetch the indices; i-dimension is bootstrap, j-dimension is component
                for i, j in np.argwhere(failures):

                    # Set it under maintenance for the correct duration
                    self.maintaining[i, idx, j] = turbines[idx].degradation_models[j].maintenance['replace']['duration']

                    # Reset D and delta_D, but keep under maintenance so it doesn't accumulate damage and costs
                    turbines[idx].degradation_models[j].reset_model(dim=i, maintenance=True)

                    # Increment the interventions
                    self.interventions += 1

            # If any components still has remaining maintenance, the turbine is not active
            # This essentially collapses the maintenance array to a (1,1) or (2,1) array with the sum over components, for a given turbine
            # If any component for that turbine had remaining maintenance time, this will evaluate to a non-zero value
            # Using .sum and a comparison is faster than .any!
            turbines[idx].active = np.sum(self.maintaining[:, idx, :], axis=1)[:, None] < 1

        self.iter += 1

        # Two possibilities: without baseline removal, we take the value as-is; with baseline removal, we take the difference
        return costs.squeeze(axis=0) if self.n_d == 1 else np.diff(-costs, axis=0).squeeze(axis=0)
    
    def __str__(self) -> str:

        return f'{self.__class__.__name__} with {np.sum(self.maintaining > 0)} turbines under maintenance:\n' + '\n'.join([f'    Bootstrap {i}, Turbine {j}, component {k} under maintenance for {self.maintaining[i, j, k]} timesteps' for i, j, k in np.argwhere(self.maintaining > 0)])
    
    def reset(self) -> None:

        self.maintaining = None
        self.running_delta_D_mean = 0
        self.iter = 0