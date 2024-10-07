
import numpy as np

from WakeWISE.DegradationModels import DeterministicFatigueModel, FatigueModel
from WakeWISE.main import SimulationEnvironment
from WakeWISE.MaintenanceModels import DenseRunToFailure, MaintenanceModel
from WakeWISE.PropagationModels import GNNSurrogate, PropagationModel
from WakeWISE.utils import (
    DataLogger,
    MonthlyHourlyRateModel,
    RateModel,
)
from WakeWISE.WindSamplers import AnholtDetailedWindSamplerV2, WindSampler
from WakeWISE.WindTurbine import IEA37Turbine, Turbine


class GradientDescentSimulationEnvironment(SimulationEnvironment):
    """
    Special class of the simulation environment. Note that this is not called as a wrapper
    but rather as a custom instance of the simulation class.
    """

    def __init__(self,
                 config: 'str | dict' = {},
                 turbine_type: 'Turbine | list[Turbine]' = IEA37Turbine,
                 degredation_model: FatigueModel = DeterministicFatigueModel,
                 wind_model: WindSampler = AnholtDetailedWindSamplerV2,
                 propagation_model: PropagationModel = GNNSurrogate,
                 maintenance_model: MaintenanceModel = DenseRunToFailure,
                 rate_model : RateModel = MonthlyHourlyRateModel,
                 deg_params = None,
                 logger: DataLogger = None,
                 debug: bool = False,
                 **kwargs) -> None:

        super().__init__(config=config,
                         turbine_type=turbine_type,
                         degredation_model=degredation_model,
                         wind_model=wind_model,
                         propagation_model=propagation_model,
                         maintenance_model=maintenance_model,
                         rate_model=rate_model,
                         deg_params=deg_params,
                         logger=logger,
                         debug=debug,
                         **kwargs)
        
        raise NotImplementedError("Uses old environment step function, needs updating")


    def fake_step(self, actions: np.ndarray) -> tuple[np.ndarray, 'float | np.ndarray', bool, bool, dict]:
        """
        Run a 'fake' step in the simulation environment, for gradient descent

        Args:
        - actions (np.ndarray): the actions to take. actions.shape = (n_turbines, 1)
        """

        # Add the actions to the operational modes
        # self.actions = self.action_normaliser.denormalize(actions) if self.normalise else actions
        turbine_states = np.hstack((actions, self.op_modes))

        # Run the propagation model
        self.results = self.model(self.wind_condition['vector'], turbine_states)

        # Small trick: we can aggregate certain DELs together and take the norm
        # The components are defined in the degradation parameters, as lists of indices
        # We can then take the norm of these components, combining them into a single value
        # If the list has only one element, we just take that value
        # Remember, order: [power, ws, ti, del_blfw, del_flew, del_tbfa, del_tbss, del_ttyaw]
        idx_lists = [self.deg_params['components'][c]['idx'] for c in self.deg_params['components']]
        self.derived_results = np.c_[[np.linalg.norm(self.results[:, i], axis=1) for i in idx_lists]].T

        # Unused, kept for reference
        # derived_results = np.zeros((self.n_turbines, len(self.deg_params['components'])))
        # for idx, component in enumerate(self.deg_params['components']):
        #     derived_results[:, idx] = np.linalg.norm(results[:, np.array(self.deg_params['components'][component]['idx'])], axis=1).T

        # Update the turbines
        for idx in self.turbines.keys():
            self.turbines[idx].update(self.derived_results[idx], self.timestep_seconds) # pass determined DELs

        # Calculate the cost from the maintenance model
        self.cost = self.maintenance_model.run(self.turbines, undo = True, **self.wind_condition)

        if  self.collective_reward:
            self.cost = np.sum(self.cost)

        # Calculate the power produced [Heck yeah vectorisation!]
        self.power_produced = self.op_modes * self.results[:, :1]

        self._get_reward()

        # Sample wind conditions and electricity rate for next timestep
        self.model.precompute_edges(self.wind_condition['vector'])
        
        for idx in self.turbines.keys():
            self.turbines[idx].undo()

        return self.reward
    
