import os

from gymnasium.envs.registration import register

from WakeWISE.main import SimulationEnvironment  # noqa: F401
from WakeWISE.utils.globals import LOG_LEVEL

os.environ['WAKEWISE_LOG_LEVEL'] = LOG_LEVEL.INFO

register(
    id='WakeWISE-v1',
    entry_point='WakeWISE:SimulationEnvironment',
)