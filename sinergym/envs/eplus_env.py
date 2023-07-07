"""
Gym environment for simulation with EnergyPlus.
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from sinergym.simulators import EnergyPlus
from queue import Queue, Empty, Full
from sinergym.config import ModelJSON
from sinergym.utils.rewards import *


class EplusEnv(gym.Env):

    metadata = {'render_modes': ['human']}

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
        meters: Dict[str, Tuple[str, str]] = {},
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
            weather_file (Union[str,List[str]]): Name of the EPW file for weather conditions. It can be specified a list of weathers files in order to sample a weather in each episode randomly.
            observation_space (gym.spaces.Box, optional): Gym Observation Space definition. Defaults to an empty observation_space (no control).
            observation_variables (List[str], optional): List with variables names in building. Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            action_variables (List[str],optional): Action variables to be controlled in building, if that actions names have not been configured manually in building, you should configure or use action_definition. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]], optional): Action mapping list for discrete actions spaces only. Defaults to empty list.
            flag_normalization (bool): Flag indicating if action space must be normalized to [-1,1]. This flag only take effect in continuous environments. Default to true.
            weather_variability (Optional[Tuple[float]], optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            action_definition (Optional[Dict[str, Any]): Dict with building components to being controlled by Sinergym automatically if it is supported. Default value to None.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """

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

        # Copy to use original variables in step.obs_dict for reward
        self.original_time_variables = time_variables
        self.original_variables = variables
        self.original_meters = meters
        self.original_actuators = actuators

        # ---------------------------------------------------------------------------- #
        #                    Define observation and action variables                   #
        # ---------------------------------------------------------------------------- #

        self.observation_variables = self.time_variables + \
            list(self.variables.keys()) + list(self.meters.keys())
        self.action_variables = list(self.actuators.keys())

        self.original_observation_variables = self.original_time_variables + \
            list(self.original_variables.keys()) + \
            list(self.original_meters.keys())
        self.original_action_variables = list(self.original_actuators.keys())

        # ---------------------------------------------------------------------------- #
        #                               Building modeling                              #
        # ---------------------------------------------------------------------------- #

        self.model = ModelJSON(
            env_name=env_name,
            json_file=self.building_file,
            weather_files=self.weather_files,
            max_ep_store=max_ep_data_store_num,
            extra_config=config_params
        )

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        # Initialized in reset method
        self.energyplus_simulation: Optional[EnergyPlus] = None

        # ---------------------------------------------------------------------------- #
        #                             Environment variables                            #
        # ---------------------------------------------------------------------------- #
        self.name = env_name
        self.episode = 0
        self.timestep = 0
        # Get environment working dir
        self.workspace_dir = self.model.experiment_path
        # Queues for E+ Python API communication
        self.obs_queue: Optional[Queue] = None
        self.info_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None
        # last obs, action and info
        self.last_obs: Optional[Dict[str, float]] = None
        self.last_info: Optional[Dict[str, float]] = None
        self.last_action: Optional[List[float]] = None

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
        self._observation_space = gym.spaces.Box(
            low=-5e6,
            high=5e6,
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
            self.action_mapping = action_mapping
            self._action_space = action_space
        # Continuous
        else:
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

        self._check_eplus_env()

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
        options = options if options is not None else self.default_options

        self.episode += 1
        self.timestep = 0

        if self.energyplus_simulation is not None:
            self.energyplus_simulation.stop()

        self.obs_queue = Queue(maxsize=1)
        self.info_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
        self.last_obs = self.observation_space.sample()
        self.last_info = {'timestep': self.timestep}

        # ------------------------ Preparation for new episode ----------------------- #

        # Get new episode working dir
        self.episode_dir = self.model.set_episode_working_dir()
        # get weather path and readapt building
        self.model.update_weather_path()
        self.model.adapt_building_to_epw()
        # Getting building, weather and Energyplus output directory
        eplus_working_building_path = self.model.save_building_model()
        eplus_working_out_path = (self.episode_dir + '/' + 'output')
        eplus_working_weather_path = self.model.apply_weather_variability(
            variation=options.get('weather_variability'))

        self.energyplus_simulation = EnergyPlus(
            building_path=eplus_working_building_path,
            weather_path=eplus_working_weather_path,
            output_path=eplus_working_out_path,
            obs_queue=self.obs_queue,
            info_queue=self.info_queue,
            act_queue=self.act_queue,
            time_variables=self.time_variables,
            variables=self.variables,
            meters=self.meters,
            actuators=self.actuators
        )

        self.energyplus_simulation.start()

        # wait for E+ warmup to complete
        if not self.energyplus_simulation.warmup_complete:
            self.energyplus_simulation.warmup_queue.get()

        # Wait to receive simulation first observation and info
        # try:
            # obs = self.obs_queue.get()
        # except Empty:
        #     obs = self.last_obs

        # Wait to receive simulation first observation and info
        try:
            obs = self.obs_queue.get()
        except Empty:
            obs = self.last_obs

        try:
            info = self.info_queue.get()
        except Empty:
            info = self.last_info

        info = self.info_queue.get()
        info.update({'timestep': self.timestep})
        self.last_obs = obs
        self.last_info = info

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
        """Sends action to the environment

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """

        # Check if action is correct for the current action space
        assert self._action_space.contains(
            action), 'Step: The action {} is not correct for the Action Space {}'.format(action, self._action_space)
        # Check if episode existed and is not terminated
        assert self.energyplus_simulation, 'Step: Environment requires to be reset before.'

        # check for simulation errors
        if self.energyplus_simulation.failed():
            print(
                f"EnergyPlus failed with {self.energyplus_simulation.sim_results['exit_code']}")
            exit(1)

        self.timestep += 1

        terminated = truncated = False

        # Get real action (action --> action_)
        action_ = self._get_action(action)

        if self.energyplus_simulation.simulation_complete:
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
                truncated = True
                obs = self.last_obs
                info = self.last_info

        # Calculate reward
        reward, rw_terms = self.reward_fn(obs)

        # Update info with
        info.update({'timestep': self.timestep,
                    'reward': reward})
        info.update(rw_terms)
        self.last_info = info

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

        self.energyplus_simulation.stop()

    # ---------------------------------------------------------------------------- #
    #                           Environment functionality                          #
    # ---------------------------------------------------------------------------- #

    def _get_action(self, action: Any) -> Union[int,
                                                float,
                                                np.integer,
                                                np.ndarray,
                                                List[Any],
                                                Tuple[Any]]:
        """Transform the action for sending it to the simulator."""

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
                          action: Union[int,
                                        float,
                                        np.integer,
                                        np.ndarray,
                                        List[Any],
                                        Tuple[Any]]) -> Union[int,
                                                              float,
                                                              np.integer,
                                                              np.ndarray,
                                                              List[Any],
                                                              Tuple[Any]]:
        """ This method transforms an action defined in gym (-1,1 in all continuous environment) action space to simulation real action space.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action received in environment

        Returns:
            Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]: Action transformed in simulator action space.
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

        self.flag_normalization = value
        self._action_space = self.normalized_space if value else self.real_space

    def _check_eplus_env(self) -> None:
        """This method checks that environment definition is correct and it has not inconsistencies.
        """
        # OBSERVATION
        assert len(self.observation_variables) == self._observation_space.shape[
            0], 'Observation space has not the same length than variable names specified.'

        # ACTION
        if self.flag_discrete:
            assert hasattr(
                self, 'action_mapping'), 'Discrete environment: action mapping should have been defined.'
            assert not hasattr(
                self, 'real_space'), 'Discrete environment: real_space should not have been defined.'
            assert not hasattr(
                self, 'normalized_space'), 'Discrete environment: normalized_space should not have been defined.'
            assert not hasattr(
                self, 'flag_normalization'), 'Discrete environment: flag_normalization should not have been defined.'
            assert self._action_space.n == len(
                self.action_mapping), 'Discrete environment: The length of the action_mapping must match the dimension of the discrete action space.'
            for values in self.action_mapping.values():
                assert len(values) == len(
                    self.action_variables), 'Discrete environment: Action mapping tuples values must have the same length than action variables specified.'
        else:
            assert len(self.action_variables) == self._action_space.shape[
                0], 'Action space shape must match with number of action variables specified.'
            assert hasattr(
                self, 'flag_normalization'), 'Continuous environment: flag_normalization attribute should have been defined.'
            assert hasattr(
                self, 'normalized_space'), 'Continuous environment: normalized_space attribute should have been defined.'
            assert hasattr(
                self, 'real_space'), 'Continuous environment: real_space attribute should have been defined.'
            assert not hasattr(
                self, 'action_mapping'), 'Continuous environment: action mapping should not have been defined.'

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #
    @property
    def action_space(
        self,
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return self._action_space

    @action_space.setter
    def action_space(self, space: gym.spaces.Space[Any]):
        self._action_space = space

    @property
    def observation_space(
        self,
    ) -> gym.spaces.Space[Any] | gym.spaces.Space[Any]:
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: gym.spaces.Space[Any]):
        self._observation_space = space
