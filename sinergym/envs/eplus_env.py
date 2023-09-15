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
from sinergym.utils.logger import Logger
from sinergym.utils.rewards import *


class EplusEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    # ---------------------------------------------------------------------------- #
    #                          Environment Terminal Logger                         #
    # ---------------------------------------------------------------------------- #

    logger = Logger().getLogger(
        name='ENVIRONMENT',
        level=LOG_ENV_LEVEL)

    # ---------------------------------------------------------------------------- #
    #                            ENVIRONMENT CONSTRUCTOR                           #
    # ---------------------------------------------------------------------------- #
    def __init__(
        self,
        building_file: str,
        weather_files: Union[str, List[str]],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(
            low=0, high=0, shape=(0,), dtype=np.float32),
        time_variables: List[str] = [],
        variables: Dict[str, Tuple[str, str]] = {},
        meters: Dict[str, str] = {},
        actuators: Dict[str, Tuple[str, str, str]] = {},
        action_mapping: Dict[int, Tuple[float, ...]] = {},
        flag_normalization: bool = True,
        weather_variability: Optional[Tuple[float, float, float]] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = {},
        max_ep_data_store_num: int = 10,
        env_name: str = 'eplus-env-v1',
        config_params: Optional[Dict[str, Any]] = None
    ):
        """Environment with EnergyPlus simulator.

        Args:
            building_file (str): Name of the JSON file with the building definition.
            weather_files (Union[str,List[str]]): Name of the EPW file for weather conditions. It can be specified a list of weathers files in order to sample a weather in each episode randomly.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            time_variables (List[str]): EnergyPlus time variables we want to observe. The name of the variable must match with the name of the E+ Data Transfer API method name. Defaults to empty list.
            variables (Dict[str, Tuple[str, str]]): Specification for EnergyPlus Output:Variable. The key name is custom, then tuple must be the original variable name and the output variable key. Defaults to empty dict.
            meters (Dict[str, str]): Specification for EnergyPlus Output:Meter. The key name is custom, then value is the original EnergyPlus Meters name.
            actuators (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Input Actuators. The key name is custom, then value is a tuple with actuator type, value type and original actuator name. Defaults to empty dict.
            action_mapping (Dict[int, Tuple[float, ...]], optional): Action mapping list for discrete actions spaces only. Defaults to empty list.
            flag_normalization (bool): Flag indicating if action space must be normalized to [-1,1]. This flag only take effect in continuous environments. Default to true.
            weather_variability (Optional[Tuple[float]], optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """

        print('#==============================================================================================#')
        self.logger.info(
            'Creating Gymnasium environment... [{}]'.format(env_name))
        print('#==============================================================================================#')

        # ---------------------------------------------------------------------------- #
        #                                     Paths                                    #
        # ---------------------------------------------------------------------------- #
        # building file
        self.building_file = building_file
        # EPW file(s) (str or List of EPW's)
        if isinstance(weather_files, str):
            self.weather_files = [weather_files]
        else:
            self.weather_files = weather_files

        # ---------------------------------------------------------------------------- #
        #                  Variables, meters and actuators definition                  #
        # ---------------------------------------------------------------------------- #

        self.time_variables = time_variables
        self.variables = variables
        self.meters = meters
        self.actuators = actuators

        # ---------------------------------------------------------------------------- #
        #                    Define observation and action variables                   #
        # ---------------------------------------------------------------------------- #

        self.observation_variables = self.time_variables + \
            list(self.variables.keys()) + list(self.meters.keys())
        self.action_variables = list(self.actuators.keys())

        # ---------------------------------------------------------------------------- #
        #                               Building modeling                              #
        # ---------------------------------------------------------------------------- #

        self.model = ModelJSON(
            env_name=env_name,
            json_file=self.building_file,
            weather_files=self.weather_files,
            actuators=self.actuators,
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
        # last obs, action and info
        self.last_obs: Optional[Dict[str, float]] = None
        self.last_info: Optional[Dict[str, float]] = None
        self.last_action: Optional[List[float]] = None

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        self.energyplus_simulator = EnergyPlus(
            name=env_name,
            obs_queue=self.obs_queue,
            info_queue=self.info_queue,
            act_queue=self.act_queue,
            time_variables=self.time_variables,
            variables=self.variables,
            meters=self.meters,
            actuators=self.actuators
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
        if 'officegrid' in self.name:
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
        # Action space type
        self.flag_discrete = (
            isinstance(
                action_space,
                gym.spaces.Discrete))

        # Discrete
        if self.flag_discrete:
            self.logger.debug('Discrete environment detected.')
            self.action_mapping = action_mapping
            self._action_space = action_space
        # Continuous
        else:
            self.logger.debug('Continuous environment detected.')
            # Defining the normalized space (always [-1,1])
            self.normalized_space = gym.spaces.Box(
                # continuous_action_def[2] --> shape
                low=np.array(
                    np.repeat(-1, action_space.shape[0]), dtype=np.float32),
                high=np.array(
                    np.repeat(1, action_space.shape[0]), dtype=np.float32),
                dtype=action_space.dtype
            )
            # Defining the real space (defined by the user in environment
            # constructor)
            self.real_space = action_space

            # Determine if action is normalized or not using flag
            self.flag_normalization = flag_normalization

            # Depending on the normalized flag, action space will be the normalized space
            # or the real space.
            if self.flag_normalization:
                self._action_space = self.normalized_space
            else:
                self._action_space = self.real_space

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
            'Environment {} created successfully.'.format(env_name))

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
            seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). if value is None, a seed will be chosen from some source of entropy. Defaults to None.
            options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
        """

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Apply options if exists, else default options
        reset_options = options if options is not None else self.default_options

        self.episode += 1
        self.timestep = 1

        # Stop oold thread of old episode if exists
        self.energyplus_simulator.stop()

        self.last_obs = self.observation_space.sample()
        self.last_info = {'timestep': self.timestep}

        # ------------------------ Preparation for new episode ----------------------- #
        print('#----------------------------------------------------------------------------------------------#')
        self.logger.info(
            'Starting a new episode... [{}] [Episode {}]'.format(
                self.name, self.episode))
        print('#----------------------------------------------------------------------------------------------#')
        # Get new episode working dir
        self.episode_dir = self.model.set_episode_working_dir()
        # get weather path
        self.model.update_weather_path()
        # Readapt building
        self.model.adapt_building_to_variables()
        self.model.adapt_building_to_meters()
        self.model.adapt_building_to_config()
        self.model.adapt_building_to_epw()
        # Getting building, weather and Energyplus output directory
        eplus_working_building_path = self.model.save_building_model()
        eplus_working_weather_path = self.model.apply_weather_variability(
            variation=reset_options.get('weather_variability'))
        eplus_working_out_path = (self.episode_dir + '/' + 'output')
        self.logger.info(
            'Saving episode output path... [{}]'.format(
                eplus_working_out_path))

        self.energyplus_simulator.start(
            building_path=eplus_working_building_path,
            weather_path=eplus_working_weather_path,
            output_path=eplus_working_out_path)

        self.logger.info('Episode {} started.'.format(self.episode))

        # wait for E+ warmup to complete
        if not self.energyplus_simulator.warmup_complete:
            self.logger.debug('Waiting for finishing WARMUP process.')
            self.energyplus_simulator.warmup_queue.get()
            self.logger.debug('WARMUP process finished.')

        # Wait to receive simulation first observation and info
        try:
            obs = self.obs_queue.get()
        except Empty:
            self.logger.warning(
                'Reset: Observation queue empty, returning a random observation (not real).')
            obs = self.last_obs

        try:
            info = self.info_queue.get()
        except Empty:
            info = self.last_info
            self.logger.warning(
                'Reset: info queue empty, returning an empty info dictionary (not real).')

        info.update({'timestep': self.timestep})
        self.last_obs = obs
        self.last_info = info

        self.logger.debug('RESET observation received: {}'.format(obs))
        self.logger.debug('RESET info received: {}'.format(info))

        return np.array(list(obs.values()), dtype=np.float32), info

    # ---------------------------------------------------------------------------- #
    #                                     STEP                                     #
    # ---------------------------------------------------------------------------- #
    def step(self,
             action: Union[int,
                           float,
                           np.integer,
                           np.ndarray,
                           List[Any],
                           Tuple[Any]]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """

        # timestep +1 and flags initialization
        self.timestep += 1
        terminated = truncated = False

        # Check if action is correct for the current action space
        try:
            assert self._action_space.contains(
                action)
        except AssertionError as err:
            self.logger.warning(
                'Step: The action {} is not correct for the Action Space {}'.format(
                    action, self._action_space))
            raise err

        # Check if episode existed and is not terminated
        try:
            assert self.energyplus_simulator
        except AssertionError as err:
            self.logger.critical(
                'Step: Environment requires to be reset before.')
            raise err

        # check for simulation errors
        try:
            assert not self.energyplus_simulator.failed()
        except AssertionError as err:
            self.logger.critical(
                'EnergyPlus failed with exit code {}'.format(
                    self.energyplus_simulator.sim_results['exit_code']))
            raise err

        # Get real action (action --> action_)
        action_ = self._get_action(action)

        if self.energyplus_simulator.simulation_complete:
            self.logger.debug(
                'Trying STEP in a simulation completed, changing TERMINATED flag to TRUE.')
            terminated = True
            obs = self.last_obs
            info = self.last_info
        else:
            # enqueue action (received by EnergyPlus through dedicated callback)
            # then wait to get next observation.
            # timeout is set to 2s to handle end of simulation cases, which happens async
            # and materializes by worker thread waiting on this queue (EnergyPlus callback
            # not consuming yet/anymore)
            # timeout value can be increased if E+ timestep takes longer
            timeout = 2
            try:
                self.act_queue.put(action_, timeout=timeout)
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                self.last_info = info = self.info_queue.get(timeout=timeout)
            except (Full, Empty):
                self.logger.debug(
                    'STEP queues not receive value, simulation must be completed.')
                terminated = True
                obs = self.last_obs
                info = self.last_info

        # Calculate reward
        reward, rw_terms = self.reward_fn(obs)

        # Update info with
        info.update({'action': action_})
        info.update({'timestep': self.timestep,
                    'reward': reward})
        info.update(rw_terms)
        self.last_info = info

        self.logger.debug('STEP observation: {}'.format(obs))
        self.logger.debug('STEP reward: {}'.format(reward))
        self.logger.debug('STEP terminated: {}'.format(terminated))
        self.logger.debug('STEP truncated: {}'.format(truncated))
        self.logger.debug('STEP info: {}'.format(info))

        return np.array(list(obs.values()), dtype=np.float32
                        ), reward, terminated, truncated, info

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
        self.logger.info('Environment closed. [{}]'.format(self.name))

    # ---------------------------------------------------------------------------- #
    #                           Environment functionality                          #
    # ---------------------------------------------------------------------------- #

    def _get_action(self, action: Any) -> List[float]:
        """Transform the action to a correct format to send it to the simulator if it is necessary.

        Args:
            action (Any): Action to be transformed

        Returns:
            List[float]: List of the action values in the correct format for the simulator
        """

        # Discrete
        if self.flag_discrete:
            # Index for action_mapping
            # Some SB3 algorithms returns array(int) in their predictions
            if isinstance(action, np.ndarray):
                action = int(action.item())
            action_ = list(self.action_mapping[action])

        # Continuous
        else:
            # Transform action to real space simulation if normalized flag is
            # true
            action_ = self._action_transform(
                action) if self.flag_normalization else action

        return action_

    def _action_transform(self,
                          action: Any) -> List[float]:
        """ This method transforms an action defined in gym (-1,1 in all continuous environment) action space to simulation real action space.

        Args:
            action (Any): Action received in environment

        Returns:
            List[float]: Action transformed in simulator real action space.
        """
        action_ = []

        for i, value in enumerate(action):
            a_max_min = self._action_space.high[i] - \
                self._action_space.low[i]
            sp_max_min = self.real_space.high[i] - \
                self.real_space.low[i]

            action_.append(
                self.real_space.low[i] +
                (
                    value -
                    self._action_space.low[i]) *
                sp_max_min /
                a_max_min)

        return action_

    def update_flag_normalization(self, value: bool) -> None:
        """Update the normalized flag in continuous environments and update the action space

        Args:
            value (bool): New flag_normalization attribute value
        """
        try:
            assert not self.flag_discrete
            self.flag_normalization = value
            self._action_space = self.normalized_space if value else self.real_space
            self.logger.debug(
                'normalization flag set up to {}, now environment action space is {}'.format(
                    self.flag_normalization, self.action_space))
        except AssertionError as err:
            self.logger.error(
                'Only continuous environments can update the normalization flag.')

    def _check_eplus_env(self) -> None:
        """This method checks that environment definition is correct and it has not inconsistencies.
        """
        # OBSERVATION
        try:
            assert len(
                self.observation_variables) == self._observation_space.shape[0]
        except AssertionError as err:
            self.logger.error(
                'Observation space has not the same length than variable names specified.')
            raise err

        # ACTION
        # Discrete
        if self.flag_discrete:
            try:
                assert hasattr(self, 'action_mapping')
            except AssertionError as err:
                self.logger.error(
                    'Discrete environment: action mapping should have been defined.')
                raise err

            try:
                assert not hasattr(self, 'real_space')
            except AssertionError as err:
                self.logger.warning(
                    'Discrete environment: real_space should not have been defined.')

            try:
                assert not hasattr(self, 'normalized_space')
            except AssertionError as err:
                self.logger.warning(
                    'Discrete environment: normalized_space should not have been defined.')

            try:
                assert not hasattr(self, 'flag_normalization')
            except AssertionError as err:
                self.logger.warning(
                    'Discrete environment: flag_normalization should not have been defined.')

            try:
                assert len(
                    self.actuators) == len(self.action_mapping[0])
            except AssertionError as err:
                self.logger.critical(
                    'Discrete environment: The length of the action_mapping must match the dimension of the controlled actuators.')
                raise err

            for values in self.action_mapping.values():
                try:
                    assert len(values) == len(self.action_variables)
                except AssertionError as err:
                    self.logger.critical(
                        'Discrete environment: Action mapping tuples values must have the same length than action variables specified.')
                    raise err
        # Continuous
        else:
            try:
                assert len(
                    self.action_variables) == self._action_space.shape[0]
            except AssertionError as err:
                self.logger.critical(
                    'Action space shape must match with number of action variables specified.')
                raise err

            try:
                assert hasattr(self, 'flag_normalization')
            except AssertionError as err:
                self.logger.error(
                    'Continuous environment: flag_normalization attribute should have been defined.')
                raise err

            try:
                assert hasattr(self, 'normalized_space')
            except AssertionError as err:
                self.logger.error(
                    'Continuous environment: normalized_space attribute should have been defined.')
                raise err

            try:
                assert hasattr(self, 'real_space')
            except AssertionError as err:
                self.logger.error(
                    'Continuous environment: real_space attribute should have been defined.')
                raise err

            try:
                assert not hasattr(self, 'action_mapping')
            except AssertionError as err:
                self.logger.warning(
                    'Continuous environment: action mapping should not have been defined.')
                raise err

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #

    # ---------------------------------- Spaces ---------------------------------- #

    @property
    def action_space(
        self
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return getattr(self, '_action_space')

    @action_space.setter
    def action_space(self, space: gym.spaces.Space[Any]):
        self._action_space = space

    @property
    def observation_space(
        self
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return getattr(self, '_observation_space')

    @observation_space.setter
    def observation_space(self, space: gym.spaces.Space[Any]):
        self._observation_space = space

    # --------------------------------- Simulator -------------------------------- #

    @property
    def var_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.var_handlers

    @property
    def meter_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.meter_handlers

    @property
    def actuator_handlers(self) -> Optional[Dict[str, int]]:
        return self.energyplus_simulator.actuator_handlers

    @property
    def available_handlers(self) -> Optional[str]:
        return self.energyplus_simulator.available_data

    @property
    def is_running(self) -> bool:
        return self.energyplus_simulator.is_running

    # ------------------------------ Building model ------------------------------ #
    @property
    def runperiod(self) -> Dict[str, int]:
        return self.model.runperiod

    @property
    def episode_length(self) -> float:
        return self.model.episode_length

    @property
    def timestep_per_episode(self) -> int:
        return self.model.timestep_per_episode

    @property
    def step_size(self) -> float:
        return self.model.step_size

    @property
    def zone_names(self) -> list:
        return self.model.zone_names

    @property
    def schedulers(self) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
        return self.model.schedulers

    # ----------------------------------- Paths ---------------------------------- #

    @property
    def workspace_path(self) -> str:
        return self.model.experiment_path

    @property
    def episode_path(self) -> str:
        return self.model.episode_path

    @property
    def building_path(self) -> str:
        return self.model.building_path

    @property
    def weather_path(self) -> str:
        return self.model.weather_path

    @property
    def ddy_path(self) -> str:
        return self.model.ddy_path

    @property
    def idd_path(self) -> str:
        return self.model.idd_path

    # -------------------------------- class print ------------------------------- #

    def info(self):
        print("""
    #==================================================================================#
                                ENVIRONMENT NAME: {}
    #==================================================================================#
    #----------------------------------------------------------------------------------#
                                ENVIRONMENT INFO:
    #----------------------------------------------------------------------------------#
    - Building file: {}
    - Zone names: {}
    - Weather file(s): {}
    - Current weather used: {}
    - Episodes executed: {}
    - Workspace directory: {}
    - Reward function: {}
    - Reset default options: {}
    - Run period: {}
    - Episode length: {}
    - Number of timesteps in an episode: {}
    - Timestep size (seconds): {}
    #----------------------------------------------------------------------------------#
                                ENVIRONMENT SPACE:
    #----------------------------------------------------------------------------------#
    - Observation space: {}
    - Observation variables: {}
    - Action space: {}
    - Action variables: {}
    - Discrete?: {}
    - Action mapping: {}
    - Normalized?: {}
    - Real action space (used in simulator): {}
    #==================================================================================#
                                    SIMULATOR
    #==================================================================================#
    *NOTE: To have information about available handlers and controlled elements, it is
    required to do env reset before to print information.*

    Is running? : {}
    #----------------------------------------------------------------------------------#
                                AVAILABLE ELEMENTS:
    #----------------------------------------------------------------------------------#
    *Some variables can not be here depending if it is defined Output:Variable field
     in building model. See documentation for more information.*

    {}
    #----------------------------------------------------------------------------------#
                                CONTROLLED ELEMENTS:
    #----------------------------------------------------------------------------------#
    - Actuators: {}
    - Variables: {}
    - Meters: {}
    - Internal Variables: None

    """.format(
            self.name,
            self.building_path,
            self.zone_names,
            self.weather_files,
            self.weather_path,
            self.episode,
            self.workspace_path,
            self.reward_fn,
            self.default_options,
            self.runperiod,
            self.episode_length,
            self.timestep_per_episode,
            self.step_size,
            self.observation_space,
            self.observation_variables,
            self.action_space,
            self.action_variables,
            self.flag_discrete,
            getattr(
                self,
                'action_mapping',
                None),
            getattr(
                self,
                'flag_normalization',
                False),
            getattr(
                self,
                'real_space',
                None),
            self.is_running,
            self.available_handlers,
            self.actuator_handlers,
            self.var_handlers,
            self.meter_handlers)
        )
