"""
Gymnasium environment for simulation with EnergyPlus.
"""

from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import yaml

from sinergym.config import ModelJSON
from sinergym.simulators import EnergyPlus
from sinergym.utils.constants import LOG_ENV_LEVEL
from sinergym.utils.logger import SimpleLogger, TerminalLogger
from sinergym.utils.rewards import *


class EplusEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    # ---------------------------------------------------------------------------- #
    #                          Environment Terminal Logger                         #
    # ---------------------------------------------------------------------------- #

    logger = TerminalLogger().getLogger(name='ENVIRONMENT', level=LOG_ENV_LEVEL)

    simple_printer = SimpleLogger().getLogger()

    # ---------------------------------------------------------------------------- #
    #                            ENVIRONMENT CONSTRUCTOR                           #
    # ---------------------------------------------------------------------------- #

    def __init__(
        self,
        building_file: str,
        weather_files: Union[str, List[str]],
        action_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=0, shape=(0,), dtype=np.float32
        ),
        time_variables: List[str] = [],
        variables: Dict[str, Tuple[str, str]] = {},
        meters: Dict[str, str] = {},
        actuators: Dict[str, Tuple[str, str, str]] = {},
        context: Dict[str, Tuple[str, str, str]] = {},
        initial_context: Optional[List[float]] = None,
        weather_variability: Optional[
            Dict[
                str,
                Union[
                    Tuple[
                        Union[float, Tuple[float, float]],
                        Union[float, Tuple[float, float]],
                        Union[float, Tuple[float, float]],
                    ],
                    Tuple[
                        Union[float, Tuple[float, float]],
                        Union[float, Tuple[float, float]],
                        Union[float, Tuple[float, float]],
                        Tuple[float, float],
                    ],
                ],
            ]
        ] = None,
        reward: Any = LinearReward,
        reward_kwargs: Dict[str, Any] = {},
        max_ep_store: int = 10,
        env_name: str = 'eplus-env-v1',
        building_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Environment with EnergyPlus simulator.

        Args:
            building_file (str): Name of the JSON file with the building definition.
            weather_files (Union[str,List[str]]): Name of the EPW file for weather conditions.
                Can also be a list of weather files to sample randomly for each episode.
            action_space (gym.spaces.Box, optional): Gym Action Space definition. Defaults to empty (no control).
            time_variables (List[str]): EnergyPlus time variables to observe. Names must match E+ Data Transfer API method names. Defaults to empty list.
            variables (Dict[str, Tuple[str, str]]): Specification for EnergyPlus Output:Variable. Key is custom name; value is tuple(original variable name, output key). Defaults to empty dict.
            meters (Dict[str, str]): Specification for EnergyPlus Output:Meter. Key is custom; value is original meter name. Defaults to empty dict.
            actuators (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Input Actuators. Key is custom; value is tuple(actuator type, value type, original name). Defaults to empty dict.
            context (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Context. Key is custom; value is tuple(actuator type, value type, original name). Used for real-time building configuration. Defaults to empty dict.
            initial_context (Optional[List[float]]): Initial context values to set in the building model. Defaults to None.
            weather_variability (Optional[Dict[str,Tuple[Union[float,Tuple[float,float]],
                                                        Union[float,Tuple[float,float]],
                                                        Union[float,Tuple[float,float]],
                                                        Optional[Tuple[float,float]]]]]): Variation for weather data for Ornstein-Uhlenbeck process.
                - sigma: standard deviation or range to sample from
                - mu: mean value or range to sample from
                - tau: time constant or range to sample from
                - var_range (optional): tuple(min_val, max_val) for clipping the variable
                Defaults to None.
            reward (Any, optional): Reward function instance. Defaults to LinearReward.
            reward_kwargs (Dict[str, Any], optional): Parameters to pass to the reward function. Defaults to empty dict.
            max_ep_store (int, optional): Number of last episode folders to store. Defaults to 10.
            env_name (str, optional): Env name for working directory generation. Defaults to 'eplus-env-v1'.
            building_config (Optional[Dict[str, Any]], optional): Extra configuration for building. Defaults to None.
            seed (Optional[int], optional): Seed for random number generator. Defaults to None.
        """

        self.simple_printer.info(
            '#==============================================================================================#'
        )
        self.logger.info('Creating Gymnasium environment.')
        self.logger.info(f'Name: {env_name}')
        self.simple_printer.info(
            '#==============================================================================================#'
        )

        # ---------------------------------------------------------------------------- #
        #                                     seed                                     #
        # ---------------------------------------------------------------------------- #
        # Set the entropy, if seed is None, a random seed will be chosen
        self.seed = seed
        np.random.seed(self.seed)

        # ---------------------------------------------------------------------------- #
        #                                     Paths                                    #
        # ---------------------------------------------------------------------------- #
        # building file
        self.building_file = building_file
        # EPW file(s) (str or List of EPW's)
        self.weather_files = weather_files

        # ---------------------------------------------------------------------------- #
        #                  Variables, meters and actuators definition                  #
        # ---------------------------------------------------------------------------- #

        self.time_variables = time_variables
        self.variables = variables
        self.meters = meters
        self.actuators = actuators
        self.context = context

        # ---------------------------------------------------------------------------- #
        #                    Define observation and action variables                   #
        # ---------------------------------------------------------------------------- #

        self.observation_variables = (
            self.time_variables + list(self.variables.keys()) + list(self.meters.keys())
        )
        self.action_variables = list(self.actuators.keys())
        self.context_variables = list(self.context.keys())

        # ---------------------------------------------------------------------------- #
        #                               Building modeling                              #
        # ---------------------------------------------------------------------------- #
        self.max_ep_store = max_ep_store
        self.building_config = building_config

        self.model = ModelJSON(
            env_name=env_name,
            json_file=self.building_file,
            weather_files=self.weather_files,
            variables=self.variables,
            meters=self.meters,
            max_ep_store=self.max_ep_store,
            building_config=self.building_config,
        )

        # ---------------------------------------------------------------------------- #
        #                             Gymnasium attributes                             #
        # ---------------------------------------------------------------------------- #
        self.name = env_name
        self.episode = 0
        self.timestep = 0
        # Queues for E+ Python API communication
        self.obs_queue = Queue(maxsize=1)
        self.info_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
        self.context_queue = Queue(maxsize=1)
        # last obs, action and info
        self.last_obs: Dict[str, float] = {}
        self.last_info: Dict[str, Any] = {}
        self.last_action: np.ndarray = np.array([], dtype=np.float32)
        self.last_context: Union[List[float], np.ndarray] = np.array(
            [], dtype=np.float32
        )

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #

        # EnergyPlus simulator
        self.energyplus_simulator = EnergyPlus(
            name=env_name,
            obs_queue=self.obs_queue,
            info_queue=self.info_queue,
            act_queue=self.act_queue,
            context_queue=self.context_queue,
            time_variables=self.time_variables,
            variables=self.variables,
            meters=self.meters,
            actuators=self.actuators,
            context=self.context,
        )

        # ---------------------------------------------------------------------------- #
        #                          reset default options                               #
        # ---------------------------------------------------------------------------- #
        self.default_options = {}
        # Weather Variability
        if weather_variability:
            self.default_options['weather_variability'] = weather_variability
        if initial_context:
            self.default_options['initial_context'] = initial_context
        # ... more reset option implementations here

        # ---------------------------------------------------------------------------- #
        #                               Observation Space                              #
        # ---------------------------------------------------------------------------- #
        # If block hardcoded for officegrid environment, will be fixed in
        # future versions
        if 'officegrid' in self.name:  # pragma: no cover
            self._observation_space = gym.spaces.Box(
                low=-6e11,
                high=6e11,
                shape=(len(time_variables) + len(variables) + len(meters),),
                dtype=np.float32,
            )
        else:
            self._observation_space = gym.spaces.Box(
                low=-5e7,
                high=5e7,
                shape=(len(time_variables) + len(variables) + len(meters),),
                dtype=np.float32,
            )

        # ---------------------------------------------------------------------------- #
        #                                 Action Space                                 #
        # ---------------------------------------------------------------------------- #
        self._action_space = action_space

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_kwargs = reward_kwargs
        self.reward_fn = reward(**reward_kwargs)

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #
        self.logger.debug('Passing the environment checker...')
        self._check_eplus_env()

        # ---------------------------------------------------------------------------- #
        #                 Save environment configuration as a YAML file                #
        # ---------------------------------------------------------------------------- #
        self.save_config()

        self.logger.info('Environment created successfully.')

    # ---------------------------------------------------------------------------- #
    #                                     RESET                                    #
    # ---------------------------------------------------------------------------- #
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). If global seed was configured in environment, reset seed will not be applied. Defaults to None.
            options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
        """

        # If global seed was configured, reset seed will not be applied.
        if not self.seed:
            np.random.seed(seed)

        # Apply options if exists, else default options
        reset_options = options if options else self.default_options

        self.episode += 1
        self.timestep = 0

        # Stop old thread of old episode if exists
        self.energyplus_simulator.stop()

        self.last_obs = dict(
            zip(self.observation_variables, self.observation_space.sample())
        )
        self.last_info = {'timestep': self.timestep}

        # ------------------------ Preparation for new episode ----------------------- #
        self.simple_printer.info(
            '#----------------------------------------------------------------------------------------------#'
        )
        self.logger.info('Starting a new episode.')
        self.logger.info(f'Episode {self.episode}: {self.name}')
        self.simple_printer.info(
            '#----------------------------------------------------------------------------------------------#'
        )
        # Get new episode working dir
        self.episode_dir = self.model.set_episode_working_dir()
        # get weather path
        self.model.update_weather_path()
        # Readapt building to epw
        self.model.adapt_building_to_epw()
        # Getting building, weather and Energyplus output directory
        eplus_working_building_path = self.model.save_building_model()
        eplus_working_weather_path = self.model.apply_weather_variability(
            weather_variability=reset_options.get('weather_variability')
        )
        eplus_working_out_path = self.episode_dir + '/' + 'output'
        self.logger.info(f'Saving episode output path in {eplus_working_out_path}.')
        self.logger.debug(f'Path: {eplus_working_out_path}')

        # Set initial context if exists
        if reset_options.get('initial_context'):
            self.update_context(reset_options['initial_context'])

        self.energyplus_simulator.start(
            building_path=eplus_working_building_path,
            weather_path=eplus_working_weather_path,
            output_path=eplus_working_out_path,
            episode=self.episode,
        )

        # wait for E+ warmup to complete
        if not self.energyplus_simulator.warmup_complete:
            self.logger.debug('Waiting for finishing WARMUP process.')
            self.energyplus_simulator.warmup_queue.get()
            self.logger.debug('WARMUP process finished.')

        # Wait to receive simulation first observation and info
        try:
            obs = self.obs_queue.get(timeout=10)
        except Empty:  # pragma: no cover
            self.logger.warning(
                'Reset: Observation queue empty, returning a random observation (not real). Probably EnergyPlus initialization is failing.'
            )
            obs = self.last_obs

        try:
            info = self.info_queue.get(timeout=10)
        except Empty:  # pragma: no cover
            info = self.last_info
            self.logger.warning(
                'Reset: info queue empty, returning an empty info dictionary (not real). Probably EnergyPlus initialization is failing.'
            )

        info.update({'timestep': self.timestep})
        self.last_obs = obs
        self.last_info = info

        self.logger.info(f'Episode {self.episode} started.')

        self.logger.debug(f'RESET observation received: {obs}')
        self.logger.debug(f'RESET info received: {info}')

        return np.fromiter(obs.values(), dtype=np.float32), info

    # ---------------------------------------------------------------------------- #
    #                                     STEP                                     #
    # ---------------------------------------------------------------------------- #
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment.

        Args:
            action (np.ndarray): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """

        # timestep +1 and flags initialization
        self.timestep += 1
        terminated, truncated = False, False

        # Check action is correct for environment
        if not self._action_space.contains(action):
            self.logger.error(
                f'Invalid action: {action} (check type is np.ndarray with np.float32 values too)'
            )
            raise ValueError(
                f'Action {action} is not valid for {
                    self._action_space} (check type is np.ndarray with np.float32 values too)'
            )

        # check for simulation errors
        if self.energyplus_simulator.failed():
            self.logger.critical(
                f'EnergyPlus failed with exit code {
                    self.energyplus_simulator.sim_results['exit_code']}'
            )
            raise RuntimeError

        if self.energyplus_simulator.simulation_complete:
            self.logger.debug(
                'Trying STEP in a simulation completed, changing TRUNCATED flag to TRUE.'
            )
            truncated = True
            obs = self.last_obs
            info = self.last_info
        else:
            # Enqueue action (received by EnergyPlus through dedicated callback)
            # then wait to get the next observation.
            # Timeout is set to 2s to handle end of simulation cases, which happens asynchronously
            # and materializes by worker thread waiting on this queue (EnergyPlus callback
            # not consuming yet/anymore).
            # Timeout value can be increased if E+ timestep takes longer.
            timeout = 2
            try:
                self.act_queue.put(action, timeout=timeout)
                self.last_action = action
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                self.last_info = info = self.info_queue.get(timeout=timeout)
            except (Full, Empty):
                self.logger.debug(
                    'STEP queues not receive value, simulation must be completed. changing TRUNCATED flag to TRUE'
                )
                truncated = True
                # No real transition happened, roll back timestep counter
                self.timestep -= 1
                obs = self.last_obs
                info = self.last_info

        # Calculate reward
        reward, rw_terms = self.reward_fn(obs)

        # Update info with
        info.update(
            {'action': action.tolist(), 'timestep': self.timestep, 'reward': reward}
        )
        info.update(rw_terms)
        self.last_info = info

        # self.logger.debug(f'STEP observation: {obs}')
        # self.logger.debug(f'STEP reward: {reward}')
        # self.logger.debug(f'STEP terminated: {terminated}')
        # self.logger.debug(f'STEP truncated: {truncated}')
        # self.logger.debug(f'STEP info: {info}')

        return (
            np.fromiter(obs.values(), dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    # ---------------------------------------------------------------------------- #
    #                                RENDER (empty)                                #
    # ---------------------------------------------------------------------------- #
    def render(self, mode: str = 'human') -> None:
        """Environment rendering.

        Args:
            mode (str, optional): Mode for rendering. Defaults to 'human'.
        """
        pass

    # ---------------------------------------------------------------------------- #
    #                                     CLOSE                                    #
    # ---------------------------------------------------------------------------- #
    def close(self) -> None:
        """End simulation."""
        self.energyplus_simulator.stop()
        self.logger.info(f'Environment closed. [{self.name}]')

    # ---------------------------------------------------------------------------- #
    #                       REAL-TIME BUILDING CONTEXT UPDATE                      #
    # ---------------------------------------------------------------------------- #
    def update_context(self, context_values: Union[np.ndarray, List[float]]) -> None:
        """Update real-time building context (actuators which are not controlled by the agent).
        Args:
            context_values (Union[np.ndarray, List[float]]): List of values to be updated in the building model.
        """
        # Check context_values consistency with context variables
        if len(context_values) != len(self.context):
            self.logger.warning(
                f'Context values must have the same length than context variables specified, and values must be in the same order. The context space is {
                    self.context}, but values {context_values} were specified.'
            )

        try:
            self.context_queue.put(context_values, block=False)
            self.last_context = context_values
        except Full:
            self.logger.warning(
                f'Context queue is full, context update with values {context_values} will be skipped.'
            )

    # ---------------------------------------------------------------------------- #
    #                           Environment functionality                          #
    # ---------------------------------------------------------------------------- #

    def save_config(self) -> None:
        """Save environment configuration as a YAML file."""
        with open(f'{self.workspace_path}/env_config.pyyaml', 'w') as f:
            yaml.dump(
                data=self.unwrapped, stream=f, default_flow_style=False, sort_keys=False
            )

    def _check_eplus_env(self) -> None:
        """This method checks that environment definition is correct and it has not inconsistencies."""
        # OBSERVATION
        assert self._observation_space.shape
        if len(self.observation_variables) != self._observation_space.shape[0]:
            self.logger.error(
                f'Observation space ({
                    self._observation_space.shape[0]} variables) has not the same length than specified variable names ({
                    len(
                        self.observation_variables)}).'
            )
            raise ValueError

        # ACTION
        assert self._action_space.shape
        if len(self.action_variables) != self._action_space.shape[0]:
            self.logger.error(
                f'Action space defined in environment( with {
                    self._action_space.shape[0]} variables) has not the same length than specified action variable names ({
                    len(
                        self.action_variables)} variables).'
            )
            raise ValueError

        # WEATHER VARIABILITY
        if 'weather_variability' in self.default_options:

            def validate_params(params):
                """Validate weather variability parameters."""
                if not isinstance(params, tuple):
                    raise ValueError(
                        f'Invalid parameter for Ornstein-Uhlenbeck process: {params}. '
                        'It must be a tuple of 3 or 4 elements.'
                    )
                if len(params) not in (3, 4):
                    raise ValueError(
                        f'Invalid parameter for Ornstein-Uhlenbeck process: {params}. '
                        'It must have exactly 3 or 4 values.'
                    )

                # Extract elements
                ou_params_dict = {
                    'sigma': params[0],
                    'mu': params[1],
                    'tau': params[2],
                }
                var_range = params[3] if len(params) == 4 else None

                # Validate sigma, mu, tau
                for ou_name, value in ou_params_dict.items():
                    if not isinstance(value, (int, float, tuple, list)):
                        raise ValueError(
                            f'Invalid {ou_name} for Ornstein-Uhlenbeck process: {value}. '
                            'It must be a number or a tuple/list of two numbers (range).'
                        )
                    if isinstance(value, (tuple, list)) and len(value) != 2:
                        raise ValueError(
                            f'Invalid {ou_name} tuple for Ornstein-Uhlenbeck process: {value}. '
                            'It must have exactly two values (range).'
                        )

                # Validate var_range if provided
                if var_range:
                    if not (
                        isinstance(var_range, (tuple, list)) and len(var_range) == 2
                    ):
                        raise ValueError(
                            f'Invalid var_range for Ornstein-Uhlenbeck process: {var_range}. '
                            'It must be a tuple/list of two numbers (min_val, max_val).'
                        )
                    if not all(isinstance(v, (int, float)) for v in var_range):
                        raise ValueError(
                            f'Invalid values in var_range: {var_range}. Both must be numbers.'
                        )

            try:
                # Validate each weather variability parameter
                for _, params in self.default_options['weather_variability'].items():
                    validate_params(params)
            except ValueError as err:
                self.logger.critical(str(err))
                raise err

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #

    def set_seed(self, seed: Optional[int]) -> None:
        """Set seed for random number generator.

        Args:
            seed (Optional[int]): Seed for random number generator.
        """
        self.seed = seed
        np.random.seed(self.seed)

    # ---------------------------------- Spaces ---------------------------------- #

    @property  # pragma: no cover
    def action_space(self) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return getattr(self, '_action_space')

    @action_space.setter  # pragma: no cover
    def action_space(self, space: gym.spaces.Space[Any]):
        self._action_space = space

    @property  # pragma: no cover
    def observation_space(self) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return getattr(self, '_observation_space')

    @observation_space.setter  # pragma: no cover
    def observation_space(self, space: gym.spaces.Space[Any]):
        self._observation_space = space

    @property  # pragma: no cover
    def is_discrete(self) -> bool:
        if isinstance(self.action_space, gym.spaces.Box):
            return False
        elif (
            isinstance(self.action_space, gym.spaces.Discrete)
            or isinstance(self.action_space, gym.spaces.MultiDiscrete)
            or isinstance(self.action_space, gym.spaces.MultiBinary)
        ):
            return True
        else:
            self.logger.warning('Action space is not continuous or discrete?')
            return False

    # --------------------------------- Simulator -------------------------------- #

    @property  # pragma: no cover
    def var_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.var_handlers

    @property  # pragma: no cover
    def meter_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.meter_handlers

    @property  # pragma: no cover
    def actuator_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.actuator_handlers

    @property  # pragma: no cover
    def context_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.context_handlers

    @property  # pragma: no cover
    def available_handlers(self) -> Optional[str]:
        return self.energyplus_simulator.available_data

    @property  # pragma: no cover
    def is_running(self) -> bool:
        return self.energyplus_simulator.is_running

    # ------------------------------ Building model ------------------------------ #
    @property  # pragma: no cover
    def runperiod(self) -> Dict[str, int]:
        return self.model.runperiod

    @property  # pragma: no cover
    def episode_length(self) -> float:
        return self.model.episode_length

    @property  # pragma: no cover
    def timestep_per_episode(self) -> int:
        return self.model.timestep_per_episode

    @property  # pragma: no cover
    def step_size(self) -> float:
        return self.model.step_size

    @property  # pragma: no cover
    def zone_names(self) -> list:
        return self.model.zone_names

    @property  # pragma: no cover
    def schedulers(self) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
        return self.model.schedulers

    # ----------------------------------- Paths ---------------------------------- #

    @property  # pragma: no cover
    def workspace_path(self) -> str:
        return self.model.workspace_path

    @property  # pragma: no cover
    def episode_path(self) -> Optional[str]:
        return self.model.episode_path

    @property  # pragma: no cover
    def building_path(self) -> str:
        return self.model.building_path

    @property  # pragma: no cover
    def weather_path(self) -> str:
        return self.model.weather_path

    @property  # pragma: no cover
    def ddy_path(self) -> Optional[str]:
        return self.model.ddy_path

    @property  # pragma: no cover
    def idd_path(self) -> Optional[str]:
        return self.model.idd_path

    # ---------------------------- formats and prints ---------------------------- #

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover
        """Convert the environment instance to a Python dictionary.

        Returns:
            Dict[str, Any]: Environment configuration.
        """
        return {
            'building_file': self.building_file,
            'weather_files': self.weather_files,
            'action_space': self.action_space,
            'time_variables': self.time_variables,
            'variables': self.variables,
            'meters': self.meters,
            'actuators': self.actuators,
            'context': self.context,
            'initial_context': self.default_options.get('initial_context'),
            'weather_variability': self.default_options.get('weather_variability'),
            'reward': self.reward_fn.__class__,
            'reward_kwargs': self.reward_kwargs,
            'max_ep_store': self.max_ep_store,
            'env_name': self.name,
            'building_config': self.building_config,
            'seed': self.seed,
        }

    @classmethod  # pragma: no cover
    def from_dict(cls, data):
        return cls(**data)

    def to_str(self):  # pragma: no cover
        print(
            f"""
    #==================================================================================#
        ENVIRONMENT NAME: {self.name}
    #==================================================================================#
    #----------------------------------------------------------------------------------#
                                    ENVIRONMENT INFO:
    #----------------------------------------------------------------------------------#
    - Building file: {self.building_path}
    - Zone names: {self.zone_names}
    - Weather file(s): {self.weather_files}
    - Current weather used: {self.weather_path}
    - Episodes executed: {self.episode}
    - Workspace directory: {self.workspace_path}
    - Reward function: {self.reward_fn}
    - Reset default options: {self.default_options}
    - Run period: {self.runperiod}
    - Episode length (seconds): {self.episode_length}
    - Number of timesteps in an episode: {self.timestep_per_episode}
    - Timestep size (seconds): {self.step_size}
    - It is discrete?: {self.is_discrete}
    - seed: {self.seed}
    #----------------------------------------------------------------------------------#
                                    ENVIRONMENT SPACE:
    #----------------------------------------------------------------------------------#
    - Observation space: {self.observation_space}
    - Observation variables: {self.observation_variables}
    - Action space: {self.action_space}
    - Action variables: {self.action_variables}
    #==================================================================================#
                                        SIMULATOR
    #==================================================================================#
    *NOTE: To have information about available handlers and controlled elements, it is
    required to do env reset before to print information.*

    Is running? : {self.is_running}
    #----------------------------------------------------------------------------------#
                                    AVAILABLE ELEMENTS:
    #----------------------------------------------------------------------------------#
    *Some variables cannot be here depending if it is defined Output:Variable field
     in building model. See documentation for more information.*
    #----------------------------------------------------------------------------------#
                                    CONTROLLED ELEMENTS:
    #----------------------------------------------------------------------------------#
    - Actuators: {self.actuator_handlers}
    - Variables: {self.var_handlers}
    - Meters: {self.meter_handlers}
    - Internal Context: {self.context_handlers}

    """
        )
