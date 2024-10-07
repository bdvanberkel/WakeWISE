import time

import numpy as np

from WakeWISE.DegradationModels import (  # noqa: F401
    DeterministicFatigueModel,
    StochasticFatigueModel,
)
from WakeWISE.main import SimulationEnvironment  # noqa: F401
from WakeWISE.MaintenanceModels import (  # noqa: F401
    DenseRunToFailure,
    SparseDoNothing,
    SparseRunToFailure,
    ExponentialDenseRunToFailureV1,
    ExponentialDenseRunToFailureV2
)
from WakeWISE.PropagationModels import GNNSurrogate
from WakeWISE.utils.DataLoggers import NPLogger  # noqa: F401
from WakeWISE.WindSamplers import (  # noqa: F401
    AnholtBasicWindSampler,
    AnholtDetailedWindSampler,
    AnholtDetailedWindSamplerV2,
    UniformSampler,
)
from WakeWISE.WindTurbine import IEA37Turbine
from WakeWISE.wrappers.DebugWrapper import DebugWrapper
from WakeWISE.wrappers.DoNothingWrapper import DoNothingWrapper
from WakeWISE.wrappers.FullyObservibleWrapper import FullyObservibleWrapper
from WakeWISE.wrappers.GraphObservationWrapper import GraphObservationWrapper
from WakeWISE.wrappers.NormalisationWrapper import ObsNormalisationWrapper
from WakeWISE.wrappers.ScaleRewardsWrapper import ScaleRewardWrapper
from WakeWISE.wrappers.RemoveBaselineWrapper import RemoveBaselineWrapper, RemoveBootstrappedBaselineWrapper

if __name__ == "__main__":

    config = {
        'render_mode': 'none'
    }

    kwargs = {}

    my_env = SimulationEnvironment(config,
                                   turbine_type=IEA37Turbine,
                                   degredation_model=DeterministicFatigueModel,
                                   wind_model=AnholtDetailedWindSamplerV2,
                                   PropagationModel=GNNSurrogate,
                                   maintenance_model=SparseDoNothing,
                                   logger=None,
                                   debug=True,
                                   **kwargs)
    
    my_env = FullyObservibleWrapper(my_env)
    my_env = GraphObservationWrapper(my_env)
    my_env = ObsNormalisationWrapper(my_env)
    my_env = RemoveBootstrappedBaselineWrapper(my_env)
    my_env = ScaleRewardWrapper(my_env)
    my_env = DebugWrapper(my_env)

    actions = np.zeros((my_env.n_turbines,))
    # actions = (np.random.random((my_env.n_turbines, 1)) - 0.5) * 2 * 5

    n_iters = 100
    n_episodes = 1

    rs = []

    start_time = time.time()

    for i in range(n_episodes):
        obs, info = my_env.reset(seed=i)

        print(i)
        r = 0

        for _ in range(n_iters):
            obs, reward, terminated, truncated, info = my_env.step(actions)

            r += np.sum(reward)

            if terminated or truncated:
                break

        rs.append(r)

    print(time.time() - start_time)
    print('Avg time per step:', (time.time() - start_time) / n_iters, 's')
    print('Avg steps per second:', 1 / ((time.time() - start_time) / n_iters), 'it/s')

    print(np.mean(rs), np.std(rs))

    my_env.close()

