"""
Gymnasium environment for simulation with EnergyPlus.
"""

from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

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

    logger = TerminalLogger().getLogger(
        name='ENVIRONMENT',
        level=LOG_ENV_LEVEL)

    simple_printer = SimpleLogger().getLogger()

    # ---------------------------------------------------------------------------- #
    #                            ENVIRONMENT CONSTRUCTOR                           #
    # ---------------------------------------------------------------------------- #
    def __init__(
        self,
        building_file: str,
        weather_files: Union[str, List[str]],
        action_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=0, shape=(0,), dtype=np.float32),
        time_variables: List[str] = [],
        variables: Dict[str, Tuple[str, str]] = {},
        meters: Dict[str, str] = {},
        actuators: Dict[str, Tuple[str, str, str]] = {},
        context: Dict[str, Tuple[str, str, str]] = {},
        initial_context: Optional[List[float]] = None,
        weather_variability: Optional[Dict[str, Tuple[
            Union[float, Tuple[float, float]],
            Union[float, Tuple[float, float]],
            Union[float, Tuple[float, float]]
        ]]] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = {},
        max_ep_data_store_num: int = 10,
        env_name: str = 'eplus-env-v1',
        config_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Environment with EnergyPlus simulator.

        Args:
            building_file (str): Name of the JSON file with the building definition.
            weather_files (Union[str,List[str]]): Name of the EPW file for weather conditions. It can be specified a list of weathers files in order to sample a weather in each episode randomly.
            action_space (gym.spaces.Box, optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            time_variables (List[str]): EnergyPlus time variables we want to observe. The name of the variable must match with the name of the E+ Data Transfer API method name. Defaults to empty list.
            variables (Dict[str, Tuple[str, str]]): Specification for EnergyPlus Output:Variable. The key name is custom, then tuple must be the original variable name and the output variable key. Defaults to empty dict.
            meters (Dict[str, str]): Specification for EnergyPlus Output:Meter. The key name is custom, then value is the original EnergyPlus Meters name.
            actuators (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Input Actuators. The key name is custom, then value is a tuple with actuator type, value type and original actuator name. Defaults to empty dict.
            context (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Context. The key name is custom, then value is a tuple with actuator type, value type and original actuator name. These values are processed as real-time building configuration instead of real-time control. Defaults to empty dict.
            initial_context (Optional[List[float]]): Initial context values to be set in the building model. Defaults to None.
            weather_variability (Optional[Dict[str,Tuple[Union[float,Tuple[float,float]],Union[float,Tuple[float,float]],Union[float,Tuple[float,float]]]]]): Tuple with sigma, mu and tau of the Ornstein-Uhlenbeck process for each desired variable to be applied to weather data. Ranges can be specified to and a value will be select randomly for each episode. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
            seed (Optional[int], optional): Seed for random number generator. Defaults to None.
        """

        self.simple_printer.info(
            '#==============================================================================================#')
        self.logger.info(
            'Creating Gymnasium environment.')
        self.logger.info(f'Name: {env_name}')
        self.simple_printer.info(
            '#==============================================================================================#')

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
        self.weather_files = [weather_files] if isinstance(
            weather_files, str) else weather_files

        # ---------------------------------------------------------------------------- #
        #                  Variables, meters and actuators definition                  #
        # ---------------------------------------------------------------------------- #

        self.time_variables = time_variables
        self.variables = variables
        self.meters = meters
        self.actuators = actuators
        self.context = context
        self.initial_context = initial_context

        # ---------------------------------------------------------------------------- #
        #                    Define observation and action variables                   #
        # ---------------------------------------------------------------------------- #

        self.observation_variables = self.time_variables + \
            list(self.variables.keys()) + list(self.meters.keys())
        self.action_variables = list(self.actuators.keys())
        self.context_variables = list(self.context.keys())

        # ---------------------------------------------------------------------------- #
        #                               Building modeling                              #
        # ---------------------------------------------------------------------------- #

        self.model = ModelJSON(
            env_name=env_name,
            json_file=self.building_file,
            weather_files=self.weather_files,
            variables=self.variables,
            meters=self.meters,
            max_ep_store=max_ep_data_store_num,
            extra_config=config_params
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
        self.last_obs: Optional[Dict[str, float]] = None
        self.last_info: Optional[Dict[str, Any]] = None
        self.last_action: Optional[np.ndarray] = None
        self.last_context: Optional[Union[List[float], np.ndarray]] = None

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #

        # Set initial context if exists
        if self.initial_context is not None:
            self.update_context(self.initial_context)

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
            context=self.context
        )

        # ---------------------------------------------------------------------------- #
        #                          reset default options                               #
        # ---------------------------------------------------------------------------- #
        self.default_options = {}
        # Weather Variability
        if weather_variability:
            self.default_options['weather_variability'] = weather_variability
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
                dtype=np.float32)
        else:
            self._observation_space = gym.spaces.Box(
                low=-5e7,
                high=5e7,
                shape=(len(time_variables) + len(variables) + len(meters),),
                dtype=np.float32)

        # ---------------------------------------------------------------------------- #
        #                                 Action Space                                 #
        # ---------------------------------------------------------------------------- #
        self._action_space = action_space

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(**reward_kwargs)

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #
        self.logger.debug('Passing the environment checker...')
        self._check_eplus_env()

        self.logger.info(
            'Environment created successfully.')

    # ---------------------------------------------------------------------------- #
    #                                     RESET                                    #
    # ---------------------------------------------------------------------------- #
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Reset the environment.

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). If global seed was configured in environment, reset seed will not be applied. Defaults to None.
            options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
        """

        # If global seed was configured, reset seed will not be applied.
        if self.seed is None:
            np.random.seed(seed)

        # Apply options if exists, else default options
        reset_options = options if options is not None else self.default_options

        self.episode += 1
        self.timestep = 0

        # Stop oold thread of old episode if exists
        self.energyplus_simulator.stop()

        self.last_obs = dict(
            zip(self.observation_variables, self.observation_space.sample()))
        self.last_info = {'timestep': self.timestep}

        # ------------------------ Preparation for new episode ----------------------- #
        self.simple_printer.info(
            '#----------------------------------------------------------------------------------------------#')
        self.logger.info(
            'Starting a new episode.')
        self.logger.info(
            f'Episode {self.episode}: {self.name}')
        self.simple_printer.info(
            '#----------------------------------------------------------------------------------------------#')
        # Get new episode working dir
        self.episode_dir = self.model.set_episode_working_dir()
        # get weather path
        self.model.update_weather_path()
        # Readapt building to epw
        self.model.adapt_building_to_epw()
        # Getting building, weather and Energyplus output directory
        eplus_working_building_path = self.model.save_building_model()
        eplus_working_weather_path = self.model.apply_weather_variability(
            weather_variability=reset_options.get('weather_variability'))
        eplus_working_out_path = (self.episode_dir + '/' + 'output')
        self.logger.info(
            f'Saving episode output path in {eplus_working_out_path}.')
        self.logger.debug(f'Path: {eplus_working_out_path}')

        self.energyplus_simulator.start(
            building_path=eplus_working_building_path,
            weather_path=eplus_working_weather_path,
            output_path=eplus_working_out_path,
            episode=self.episode)

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
                'Reset: Observation queue empty, returning a random observation (not real). Probably EnergyPlus initialization is failing.')
            obs = self.last_obs

        try:
            info = self.info_queue.get(timeout=10)
        except Empty:  # pragma: no cover
            info = self.last_info
            self.logger.warning(
                'Reset: info queue empty, returning an empty info dictionary (not real). Probably EnergyPlus initialization is failing.')

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
    def step(self,
             action: np.ndarray
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
                f'Invalid action: {action} (check type is np.ndarray with np.float32 values too)')
            raise ValueError(
                f'Action {action} is not valid for {
                    self._action_space} (check type is np.ndarray with np.float32 values too)')

        # check for simulation errors
        if self.energyplus_simulator.failed():
            self.logger.critical(
                f'EnergyPlus failed with exit code {
                    self.energyplus_simulator.sim_results['exit_code']}')
            raise RuntimeError

        if self.energyplus_simulator.simulation_complete:
            self.logger.debug(
                'Trying STEP in a simulation completed, changing TRUNCATED flag to TRUE.')
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
                self.last_info = info = self.info_queue.get(
                    timeout=timeout)
            except (Full, Empty):
                self.logger.debug(
                    'STEP queues not receive value, simulation must be completed. changing TRUNCATED flag to TRUE')
                truncated = True
                obs = self.last_obs
                info = self.last_info

        # Calculate reward
        reward, rw_terms = self.reward_fn(obs)

        # Update info with
        info.update({'action': action.tolist(),
                     'timestep': self.timestep,
                     'reward': reward})
        info.update(rw_terms)
        self.last_info = info

        # self.logger.debug(f'STEP observation: {obs}')
        # self.logger.debug(f'STEP reward: {reward}')
        # self.logger.debug(f'STEP terminated: {terminated}')
        # self.logger.debug(f'STEP truncated: {truncated}')
        # self.logger.debug(f'STEP info: {info}')

        return np.fromiter(
            obs.values(), dtype=np.float32), reward, terminated, truncated, info

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
    def update_context(self,
                       context_values: Union[np.ndarray,
                                             List[float]]) -> None:
        """Update real-time building context (actuators which are not controlled by the agent).

        Args:
            context_values (Union[np.ndarray, List[float]]): List of values to be updated in the building model.
        """
        # Check context_values concistency with context variables
        if len(context_values) != len(self.context):
            self.logger.warning(
                f'Context values must have the same length than context variables specified, and values must be in the same order. The context space is {
                    self.context}, but values {context_values} were spevified.')

        try:
            self.context_queue.put(context_values, block=False)
            self.last_context = context_values
        except (Full):
            self.logger.warning(
                f'Context queue is full, context update with values {context_values} will be skipped.')

    # ---------------------------------------------------------------------------- #
    #                           Environment functionality                          #
    # ---------------------------------------------------------------------------- #

    def _check_eplus_env(self) -> None:
        """This method checks that environment definition is correct and it has not inconsistencies.
        """
        # OBSERVATION
        if len(self.observation_variables) != self._observation_space.shape[0]:
            self.logger.error(
                f'Observation space ({
                    self._observation_space.shape[0]} variables) has not the same length than specified variable names ({
                    len(
                        self.observation_variables)}).')
            raise ValueError

        # ACTION
        if len(self.action_variables) != self._action_space.shape[0]:
            self.logger.error(
                f'Action space ({
                    self._action_space.shape[0]} variables) has not the same length than specified variable names ({
                    len(
                        self.action_variables)}).')
            raise ValueError

        # WEATHER VARIABILITY
        if 'weather_variability' in self.default_options:
            def validate_params(params):
                """Validate weather variability parameters."""
                if not (isinstance(params, tuple)):
                    raise ValueError(
                        f'Invalid parameter for Ornstein-Uhlenbeck process: {
                            params}. '
                        'It must be a tuple of 3 elements.'
                    )
                if len(params) != 3:
                    raise ValueError(
                        f'Invalid parameter for Ornstein-Uhlenbeck process: {
                            params}.'
                        'It must have exactly 3 values.'
                    )

                for param in params:
                    if not (
                        isinstance(
                            param,
                            (tuple, float, int))
                    ):
                        raise ValueError(
                            f'Invalid parameter for Ornstein-Uhlenbeck process: {
                                param}. '
                            'It must be a tuple of two values (range), or a number.'
                        )
                    if (isinstance(param, tuple)) and len(param) != 2:
                        raise ValueError(
                            f'Invalid parameter for Ornstein-Uhlenbeck process: {
                                param}. '
                            'Tuples must have exactly two values (range).'
                        )

            try:
                # Validate each weather variability parameter
                for _, params in self.default_options['weather_variability'].items(
                ):
                    validate_params(params)
            except ValueError as err:
                self.logger.critical(str(err))  # Convert the error to a string
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
    def action_space(
        self
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return getattr(self, '_action_space')

    @action_space.setter  # pragma: no cover
    def action_space(self, space: gym.spaces.Space[Any]):
        self._action_space = space

    @property  # pragma: no cover
    def observation_space(
        self
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return getattr(self, '_observation_space')

    @observation_space.setter  # pragma: no cover
    def observation_space(self, space: gym.spaces.Space[Any]):
        self._observation_space = space

    @property  # pragma: no cover
    def is_discrete(self) -> bool:
        if isinstance(self.action_space, gym.spaces.Box):
            return False
        elif isinstance(self.action_space, gym.spaces.Discrete) or \
                isinstance(self.action_space, gym.spaces.MultiDiscrete) or \
                isinstance(self.action_space, gym.spaces.MultiBinary):
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
        return self.model.experiment_path

    @property  # pragma: no cover
    def episode_path(self) -> str:
        return self.model.episode_path

    @property  # pragma: no cover
    def building_path(self) -> str:
        return self.model.building_path

    @property  # pragma: no cover
    def weather_path(self) -> str:
        return self.model.weather_path

    @property  # pragma: no cover
    def ddy_path(self) -> str:
        return self.model.ddy_path

    @property  # pragma: no cover
    def idd_path(self) -> str:
        return self.model.idd_path

    # -------------------------------- class print ------------------------------- #

    def info(self):  # pragma: no cover
        print(f"""
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

    """)
