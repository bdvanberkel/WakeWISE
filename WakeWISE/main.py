from pathlib import Path

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box

from WakeWISE.DegradationModels import DeterministicFatigueModel, FatigueModel
from WakeWISE.MaintenanceModels import DenseRunToFailure, MaintenanceModel
from WakeWISE.PropagationModels import GNNSurrogate, PropagationModel
from WakeWISE.utils import (
    DataLogger,
    MonthlyHourlyRateModel,
    RateModel,
    TimeKeeper,
    get_config,
)
from WakeWISE.utils.logger import log
from WakeWISE.utils.PyGameUtils import draw_arrow, pywake_to_math_rad, value_map
from WakeWISE.WindSamplers import AnholtDetailedWindSamplerV2, WindSampler
from WakeWISE.WindTurbine import IEA37Turbine, Turbine

base_path = Path(__file__).parent

class g:
    # Global variables for the environment
    # Using mutable object to allow for global variables to be updated across modules
    # Simplifies the scheduled parameters because we can just update the global variable
    # instead of passing it around.
    TIMESTEP = 0
    EPISODE = 0
    FRACTION = 0

class SimulationEnvironment(Env):
    """
    Simulation Environment for a wake steering wind farm
    Considers the following components:
    - Wind conditions
    - Turbine loads
    - Power production
    - Maintenance costs
    - Profit
    """

    metadata = {"render_modes": ["human", "text", "machine"], "render_fps": 60}

    def __init__(self,
                 config: 'str | dict' = {},
                 turbine_type: 'Turbine | list[Turbine]' = IEA37Turbine,
                 degredation_model: FatigueModel = DeterministicFatigueModel,
                 wind_model: WindSampler = AnholtDetailedWindSamplerV2,
                 propagation_model: PropagationModel = GNNSurrogate,
                 maintenance_model: MaintenanceModel = DenseRunToFailure,
                 rate_model : RateModel = MonthlyHourlyRateModel,
                 deg_params : 'str | dict' = None,
                 logger: DataLogger = None,
                 **kwargs):
        """
        Initialises an instance of the simulation environment
        
        Args:
            - config (str | dict): the configuration for the simulation environment. If a string is given, it is assumed to be a path to a JSON file
            - turbine_type (Turbine | list[Turbine]): the type of turbine to use. If a list is given, the length must match the number of turbines in the wind farm
            - degredation_model (FatigueModel): the degradation model to use.
            - wind_model (WindSampler): the wind model to use.
            - propagation_model (PropagationModel): the propagation model to use
            - maintenance_model (MaintenanceModel): the maintenance model to use
            - rate_model (RateModel): the electricity rate model to use
            - deg_params (str): the filepath of degradation parameters to use
            - logger (DataLogger): the logger to use
            - debug (bool): whether to print debug information
            - **kwargs: additional arguments
        """

        super().__init__()

        self.deg_params = get_config(deg_params or f'{base_path}/configs/degradation.yaml')
        default_config = get_config(f'{base_path}/configs/environment.yaml')

        kwargs = {**default_config, **config, **self.deg_params, **kwargs}

        self.model : PropagationModel = propagation_model(**kwargs)
        self.n_turbines: int = self.model.n_turbines
        self.n_components: int = len(self.deg_params['components'])
        self.collective_observation: bool = kwargs.get('multi_agent_collective_observation', False)
        self.collective_reward: bool = kwargs.get('multi_agent_collective_reward', False)
        self.timestep: int = kwargs.get('timestep', 10)
        self.timestep_seconds: int = self.timestep * 60
        self.max_timesteps: int = kwargs.get('max_episode_timesteps', 1e3)
        self.random_seasonal_time: bool = kwargs.get('random_seasonal_time', True)
        self.random_windfarm_time: bool = kwargs.get('random_windfarm_time', True)
        self.seasonal_time: TimeKeeper = TimeKeeper()
        self.windfarm_time: TimeKeeper = TimeKeeper(**kwargs)
        self.wind_model : WindSampler = wind_model(kwargs.get('wind_params', {}), self.timestep, self.seasonal_time)
        self.maintenance_model : MaintenanceModel = maintenance_model(g, **{**self.deg_params['extra'], **kwargs['maintenance']})
        self.rate_model : RateModel = rate_model()
        self.render_mode: bool = kwargs.get('render_mode', None)
        self.reward_mode: str = kwargs.get('reward_mode', 'combined')
        self.debug: bool = kwargs.get('debug', False)
        self.g = g

        # Create the turbines
        if isinstance(turbine_type, list):
            assert len(turbine_type) == self.n_turbines, 'Error: Number of turbines and number of turbine types do not match'
            self.turbines : dict[int: Turbine] = {idx: turbine_type[idx](idx, degredation_model, self.deg_params, **kwargs) for idx in range(self.n_turbines)}
        else:
            self.turbines : dict[int: Turbine] = {idx: turbine_type(idx, degredation_model, self.deg_params, **kwargs) for idx in range(self.n_turbines)}

        if logger:
            self.logger: DataLogger = logger(**kwargs)

        if self.render_mode == "human":
            self.__init_anim(**kwargs)

        # Initialise the action/observation spaces
        self._define_spaces()

        # Prime the environment
        self.reset(kwargs.get('seed', None))

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, 'float | np.ndarray', bool, bool, dict]:
        """
        Run a step in the simulation environment

        Args:
        - actions (np.ndarray): the actions to take. actions.shape = (n_turbines, 1) OR (n_turbines, 2) for baseline removal

        Returns:
        - obs (dict | tuple | np.ndarray): the observation from the environment
        - reward (float | np.ndarray): the reward from the environment; float (collective_reward = True) or np.ndarray (collective_reward = False)
        - terminated (bool): whether the simulation is terminated; takes precedence over truncation
        - truncated (bool): whether the simulation is truncated
        - info (dict): additional information
        """

        # If window is closed, end the simulation
        if self.render_mode == 'human' and pygame.event.get(pygame.QUIT):
            self.close()
            return None, 0, True, False, {}
        
        # Set attribute so it can be used in observations later
        # If no baseline removal, actions has shape (n_turbines, 1)
        # If baseline removal, actions has shape (n_turbines, 2)
        self.actions = actions.reshape((-1, 1)) if actions.ndim == 1 else actions.T

        # Update the timestep (iteration) counter
        g.TIMESTEP += 1

        # For debugging purposes
        if self.debug and g.TIMESTEP % 1000 == 0:
            log(f'Iteration {g.TIMESTEP}', 'handler.py')

        # Add operational modes to the turbine states
        # If no baseline removal, the array becomes (1, n_turbines, 2); yaw and op_modes along third dimension
        # If baseline removal, the array becomes (2, n_turbines, 2); yaw and op_modes along third dimension
        turbine_states = np.array([np.concatenate((self.actions[:, i, None], self.op_modes[i, :]), axis=1) for i in range(self.actions.shape[1])])

        # Run the propagation model
        # Results has shape (1, n_turbines, n_outputs) or (2, n_turbines, n_outputs)
        # depending on whether baseline removal is used
        self.results = self.model(self.wind_condition['vector'], turbine_states)

        # Small trick: we can aggregate certain DELs together and take the norm
        # The components are defined in the degradation parameters, as lists of indices
        # We can then take the norm of these components, combining them into a single value
        # If the list has only one element, we just take that value
        # Remember, order (for default surrogate): [power, ws, ti, del_blfw, del_flew, del_tbfa, del_tbss, del_ttyaw]
        idx_lists = [self.deg_params['components'][c]['idx'] for c in self.deg_params['components']]
        self.derived_results = np.c_[[np.linalg.norm(self.results[:, :, i], axis=2).T for i in idx_lists]].T

        # derived_results has shape (1, n_turbines, n_components) or (2, n_turbines, n_components) depending on baseline removal
        self.derived_results = np.maximum(self.derived_results, 1) # Clip to prevent division by zero; np.maximum is faster than np.clip for single-sided clipping

        #####################################################################################################################################
        # Unused, kept for reference                                                                                                        #
        # derived_results = np.zeros((self.n_turbines, len(self.deg_params['components'])))                                                 #
        # for idx, component in enumerate(self.deg_params['components']):                                                                   #
        #     derived_results[:, idx] = np.linalg.norm(results[:, np.array(self.deg_params['components'][component]['idx'])], axis=1).T     #
        #####################################################################################################################################

        # Update the turbines
        for idx in self.turbines.keys():
            # pass derived results, will be of shape (1, n_components) or (2, n_components) depending on baseline removal
            self.turbines[idx].update(self.derived_results[:, idx], self.timestep_seconds) # pass determined DELs

        # Calculate the cost from the maintenance model
        # Returns costs in shape (n_turbines, 1), regardless of baseline removal (if removal, difference is calculated in module)
        self.cost = self.maintenance_model.run(self.turbines, **self.wind_condition)

        # If we want collective rewards (e.g. single reward value for all turbines combined), sum the costs
        if self.collective_reward:
            self.cost = np.sum(self.cost)

        # Calculate the power produced [Heck yeah vectorisation!]
        # Now, op_modes is of shape (1, n_turbines, 1) or (2, n_turbines, 1)
        # Results can be of shape (1, n_turbines, n_outputs) or (2, n_turbines, n_outputs)
        # Multiply powers with op modes before calculating relative power
        self.raw_power = self.op_modes[:self.results.shape[0]] * self.results[:, :, :1]
        self.power_produced = self.raw_power.squeeze(axis=0) if self.results.shape[0] == 1 else np.diff(-self.raw_power, axis=0).squeeze(axis=0)

        # Determine the new operational modes as an array of shape (1, n_turbines, 1) or (2, n_turbines, 1) depending on baseline removal
        self.op_modes = np.concatenate([self.turbines[t].active[:, :, None] for t in self.turbines], axis=1).astype(int)

        # Calculate the reward
        reward = self._get_reward()

        # Render the frame
        self.render()

        # Update the time
        terminated: bool = self.windfarm_time.update(minutes=self.timestep)
        self.seasonal_time.update(minutes=self.timestep)

        # Sample wind conditions and electricity rate for next timestep
        self.wind_condition : dict = self.wind_model.step()
        self.rate : float = self.rate_model(self.seasonal_time)

        # Precompute the edge features; allows us to use the edge features in the observation for graph-based models
        self.model.precompute_edges(self.wind_condition['vector'])

        # We can truncate if the maximum number of timesteps is reached
        truncated : bool = g.TIMESTEP >= self.max_timesteps

        # Termination takes precedence over truncation
        truncated : bool = truncated and not terminated

        info : dict = {'wind': self.wind_condition}
        obs : 'dict | tuple | np.ndarray' = self._get_observation()

        if hasattr(self, 'logger'):
            self.logger(self.wind_condition, self.turbines, self.results, self.rate, actions, self.op_modes)

        return obs, reward, terminated, truncated, info
    
    def _get_reward(self) -> float:
        """
        Calculate the reward for the environment. Considers profit from power production and maintenance costs

        Returns:
        - reward (float): the reward for the environment
        """

        # Rate is EUR per MWh; model output is in kW
        # First, we calculate the energy produced in J
        # Next, we need to convert the energy produced to MWh: 1 MWh = 3_600_000_000 J
        # Then we multiply by the rate to get the profit
        self.profit = ((self.power_produced * self.timestep_seconds) / 3_600_000) * self.rate

        # If single-agent environment, sum the profits
        if self.collective_reward:
            self.profit = np.sum(self.profit)

        # Based on reward mode, determine the reward
        if self.reward_mode == 'profit':
            self.reward = self.profit
        elif self.reward_mode == 'cost':
            self.reward = -self.cost
        elif self.reward_mode == 'combined':
            self.reward = self.profit - self.cost
        else:
            # Bruh can you read, it gives three options in the config file. Choose one of them
            raise ValueError(f'Error: Reward mode {self.reward_mode} not recognised')

        return self.reward
    
    def _get_observation(self) -> 'dict | tuple | np.ndarray':
        """
        Get the observation for the environment

        Returns:
        - obs (dict | tuple | ndarray): the observation for the environment
        """

        if self.collective_observation:
            obs = np.hstack((self.wind_condition['ws'], self.wind_condition['wd'], self.wind_condition['ti'], self.wind_condition['alpha'], self.rate, self.results[0, :, 3:].flatten())).astype(np.float32)
        else:
            obs = np.hstack(([[self.wind_condition['ws'], self.wind_condition['wd'], self.wind_condition['ti'], self.wind_condition['alpha'], self.rate]] * self.results.shape[1], self.results[0, :, 3:])).astype(np.float32)

        return obs

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        """
        Reset the environment

        Args:
        - seed (int): the seed for the environment
        - options (dict): additional options

        Returns:
        - obs (dict | tuple | np.ndarray): the observation for the reset environment
        """

        # super().reset(seed=seed) # Not sure if this is necessary; not using the gymnasium RNG instance
        np.random.seed(seed)

        # Reset the turbines
        for turbine in self.turbines:
            self.turbines[turbine].reset()

        self.maintenance_model.reset()
        self.seasonal_time.reset(random=self.random_seasonal_time)
        self.windfarm_time.reset(random=self.random_windfarm_time)
        self.wind_condition : dict = self.wind_model.reset()
        self.rate = self.rate_model(self.seasonal_time)
        self.power_produced : float = 0
        self.model.precompute_edges(self.wind_condition['vector'])

        # Reset the iteration counter
        g.TIMESTEP = 0
        g.EPISODE += 1
        g.FRACTION = self.windfarm_time.get_lifetime_fraction()

        # Initialise empty tensors
        self.op_modes : np.ndarray = np.ones((2, self.n_turbines, 1), dtype=int) # add the extra dimension just in case. Later on its inferred from data
        self.results : np.ndarray = np.zeros((1, self.n_turbines, self.model.n_outputs))
        self.actions : np.ndarray = np.zeros((self.n_turbines, 1))

        obs : dict = self._get_observation()
        info : dict = {'wind': self.wind_condition}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def __str__(self) -> str:
        return f'Environment at {str(self.seasonal_time)} with {len(self.turbines)} turbines after {self.maintenance_model.interventions} maintenance interventions:\n' + '\n'.join(['    ' + str(self.turbines[idx]) for idx in self.turbines])

    def render(self):
            
        if self.render_mode == 'text':
            print(self)
        elif self.render_mode == "human" or self.render_mode == "machine":
            self._render_frame()
        else:
            raise ValueError(f'Error: Render mode {self.render_mode} not recognised')
        
    def _render_frame(self):

        # Initialise display if we are rendering to screen
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont('Courier New', 15)

        # Initialise clock if we are rendering to screen
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.ocean_color)

        # Create text to render to canvas; information about environment
        wd_text = self.font.render(f'Wind Direction: {self.wind_condition["wd"]:>10.2f} deg', True, (0, 0, 0))
        ws_text = self.font.render(f'Wind Speed: {self.wind_condition["ws"]:>10.2f} m/s', True, (0, 0, 0))
        pw_text = self.font.render(f'Power Produced: {np.sum(self.power_produced)/1000:>10.0f} MW ', True, (0, 0, 0))
        ts_text = self.font.render(f'Timestep: {g.TIMESTEP:>10.0f} (-)', True, (0, 0, 0))
        dt_text = self.font.render(f'{str(self.seasonal_time):>10}    ', True, (0, 0, 0))
        fps_text = self.font.render(f'FPS: {self.clock.get_fps():>10.2f}', True, (0, 0, 0, 125))

        # Draw text; top right of screen going down
        canvas.blit(wd_text, (self.window_size - wd_text.get_width() - 20, 20))
        canvas.blit(ws_text, (self.window_size - ws_text.get_width() - 20, 40))
        canvas.blit(pw_text, (self.window_size - pw_text.get_width() - 20, 60))
        canvas.blit(ts_text, (self.window_size - ts_text.get_width() - 20, 80))
        canvas.blit(dt_text, (self.window_size - dt_text.get_width() - 20, 100))
        canvas.blit(fps_text, (self.window_size - fps_text.get_width() - 20, self.window_size - 20))

        # Create arrow for wind direction / wind speed
        length = value_map(0, 30, 10, 40, self.wind_condition['ws']) / 2
        center = pygame.Vector2(40, 40)
        start = pygame.Vector2(center.x + length * np.cos(pywake_to_math_rad(self.wind_condition['wd'])), center.y - length * np.sin(pywake_to_math_rad(self.wind_condition['wd'])))
        end = pygame.Vector2(center.x - length * np.cos(pywake_to_math_rad(self.wind_condition['wd'])), center.y + length * np.sin(pywake_to_math_rad(self.wind_condition['wd'])))
        draw_arrow(canvas, start, end, 0, body_width=2, head_width=6, head_height=4)

        # Draw turbines
        self.turb_pos = self.model.get_positions()
        for i in range(self.n_turbines):

            # Get coordinates of turbine
            lat, lon = self.turb_pos[i, :]

            # Remap to desired range on screen
            x, y = value_map(min(self.turb_pos[:, 0]), max(self.turb_pos[:, 0]), 0.15 * self.window_size, 0.85 * self.window_size, lat), value_map(min(self.turb_pos[:, 1]), max(self.turb_pos[:, 1]), 0.2 * self.window_size, 0.90 * self.window_size, lon)
            
            # Draw simple lines as turbines
            if self.graphics == 'simple':
                angle = pywake_to_math_rad(self.wind_condition['wd'] + self.actions[i, 0] + 90)
                x1, y1 = x - 10 * np.cos(angle), y - 10 * np.sin(angle)
                x2, y2 = x + 10 * np.cos(angle), y + 10 * np.sin(angle)
                pygame.draw.line(canvas, 0, pygame.Vector2(x1, y1), pygame.Vector2(x2, y2), width=2)
                pygame.draw.circle(canvas, 0, pygame.Vector2(x, y), 4)

            # Draw fancy turbine sprites
            elif self.graphics == 'fancy':
                angle = -(self.wind_condition['wd'] + self.actions[i, 0]) - 90
                wind_turbine_sprite_i = pygame.transform.rotate(self.wind_turbine_sprite, angle)
                canvas.blit(wind_turbine_sprite_i, wind_turbine_sprite_i.get_rect(center = (x, y)))

        # If we are rendering to screen, blit it to the window
        if self.render_mode == "human":

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        # Otherwise, return it as tensor to be stored somewhere
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def _define_spaces(self):
        """
        Define the observation and action spaces for the environment
        """

        # ws, wd, ti, rate, loads_ij
        if self.collective_observation:
            lower_bounds = np.array([0, 0, 0, self.wind_model.alpha_min_limit, self.rate_model.min] + [0, 0, 0, 0, 0] * self.n_turbines, dtype=np.float32)
            upper_bounds = np.array([30, 360, 0.5, self.wind_model.alpha_max_limit, self.rate_model.max] + [12000, 5500, 6500, 35000, 40000] * self.n_turbines, dtype=np.float32)
            self.observation_space = Box(low=lower_bounds, high=upper_bounds, shape=(5 + 5 * self.n_turbines,), dtype=np.float32)
        else:
            lower_bounds = np.array([[0, 0, 0, self.wind_model.alpha_min_limit, self.rate_model.min, 0, 0, 0, 0, 0]] * self.n_turbines, dtype=np.float32)
            upper_bounds = np.array([[30, 360, 0.5, self.wind_model.alpha_max_limit, self.rate_model.max, 12000, 5500, 6500, 35000, 40000]] * self.n_turbines, dtype=np.float32)
            self.observation_space = Box(low=lower_bounds, high=upper_bounds, shape=(self.n_turbines, 10), dtype=np.float32)

        self.action_space = Box(low=-30., high=+30., shape=(self.n_turbines,), dtype=np.float32)
    
    def close(self):
        """
        Close the environment; close any open windows and running processes
        """

        if hasattr(self, 'logger'):
            self.logger.end()

        if self.render_mode == 'human' and self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def __init_anim(self, **kwargs):
        """
        Initialise the animation window

        Args:
        - kwargs (dict): additional arguments
        """

        self.window = None
        self.clock = None
        self.turb_pos = self.model.get_positions()
        self.window_size = kwargs.get('window_size', 800)
        self.graphics = kwargs.get('graphics', 'fancy')

        assert self.graphics in ['fancy', 'simple'], f'Error: Graphics mode {self.graphics} not recognised'

        self.ocean_color = (29,162,216)

        if self.graphics == 'fancy':
            scale = 0.3
            self.wind_turbine_sprite = pygame.image.load(f'{base_path}/assets/WindTurbine.png')
            self.wind_turbine_sprite = pygame.transform.scale(self.wind_turbine_sprite, (int(self.wind_turbine_sprite.get_width() * scale), int(self.wind_turbine_sprite.get_height() * scale)))