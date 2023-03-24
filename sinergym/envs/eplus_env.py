"""
Gym environment for simulation with EnergyPlus.
"""

import os
import random
from sqlite3 import DatabaseError
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from sinergym.simulators import EnergyPlus
from sinergym.utils.common import export_actuators_to_excel
from sinergym.utils.rewards import ExpReward, LinearReward


class EplusEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    # ---------------------------------------------------------------------------- #
    #                            ENVIRONMENT CONSTRUCTOR                           #
    # ---------------------------------------------------------------------------- #
    def __init__(
        self,
        idf_file: str,
        weather_file: Union[str, List[str]],
        observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(4,), dtype=np.float32),
        observation_variables: List[str] = [],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(
            low=0, high=0, shape=(0,), dtype=np.float32),
        action_variables: List[str] = [],
        action_mapping: Dict[int, Tuple[float, ...]] = {},
        weather_variability: Optional[Tuple[float]] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = {},
        act_repeat: int = 1,
        max_ep_data_store_num: int = 10,
        action_definition: Optional[Dict[str, Any]] = None,
        env_name: str = 'eplus-env-v1',
        config_params: Optional[Dict[str, Any]] = None
    ):
        """Environment with EnergyPlus simulator.

        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (Union[str,List[str]]): Name of the EPW file for weather conditions. It can be specified a list of weathers files in order to sample a weather in each episode randomly.
            observation_space (gym.spaces.Box, optional): Gym Observation Space definition. Defaults to an empty observation_space (no control).
            observation_variables (List[str], optional): List with variables names in IDF. Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            action_variables (List[str],optional): Action variables to be controlled in IDF, if that actions names have not been configured manually in IDF, you should configure or use extra_config. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]], optional): Action mapping list for discrete actions spaces only. Defaults to empty list.
            weather_variability (Optional[Tuple[float]], optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            act_repeat (int, optional): Number of timesteps that an action is repeated in the simulator, regardless of the actions it receives during that repetition interval.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            action_definition (Optional[Dict[str, Any]): Dict with building components to being controlled by Sinergym automatically if it is supported. Default value to None.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """

        # ---------------------------------------------------------------------------- #
        #                          Energyplus, BCVTB and paths                         #
        # ---------------------------------------------------------------------------- #
        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']

        # IDF file
        self.idf_file = idf_file
        # EPW file(s) (str or List of EPW's)
        if isinstance(weather_file, str):
            self.weather_files = [weather_file]
        else:
            self.weather_files = weather_file

        # ---------------------------------------------------------------------------- #
        #                             Variables definition                             #
        # ---------------------------------------------------------------------------- #
        self.variables = {}
        self.variables['observation'] = observation_variables
        self.variables['action'] = action_variables

        self.name = env_name

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        self.simulator = EnergyPlus(
            env_name=env_name,
            eplus_path=eplus_path,
            bcvtb_path=bcvtb_path,
            idf_file=self.idf_file,
            weather_files=self.weather_files,
            variables=self.variables,
            act_repeat=act_repeat,
            max_ep_data_store_num=max_ep_data_store_num,
            action_definition=action_definition,
            config_params=config_params
        )

        # ---------------------------------------------------------------------------- #
        #                       Detection of controllable planners                     #
        # ---------------------------------------------------------------------------- #
        self.schedulers = self.get_schedulers()

        # ---------------------------------------------------------------------------- #
        #        Adding simulation date to observation (not needed in simulator)       #
        # ---------------------------------------------------------------------------- #

        self.variables['observation'] = ['year', 'month',
                                         'day', 'hour'] + self.variables['observation']

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
        self._observation_space = observation_space

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
            # Defining action values setpoints (one per value)
            self.setpoints_space = action_space

            self._action_space = gym.spaces.Box(
                # continuous_action_def[2] --> shape
                low=np.array(
                    np.repeat(-1, action_space.shape[0]), dtype=np.float32),
                high=np.array(
                    np.repeat(1, action_space.shape[0]), dtype=np.float32),
                dtype=action_space.dtype
            )

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None

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

        # Change to next episode
        # if no options specified and environment has default reset options
        if not options and len(self.default_options) > 0:
            obs, info = self.simulator.reset(
                self.default_options)
        else:
            obs, info = self.simulator.reset(
                options)

        return np.array(obs, dtype=np.float32), info

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

        # Get action
        action_ = self._get_action(action)
        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        # Execute action in simulation
        obs, terminated, truncated, info = self.simulator.step(action_)
        # Create dictionary with observation
        self.obs_dict = dict(zip(self.variables['observation'], obs))

        # Calculate reward
        reward, terms = self.reward_fn()

        # info update with reward information
        info.update({'reward': reward})
        info.update(terms)

        return np.array(
            obs, dtype=np.float32), reward, terminated, truncated, info

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

        self.simulator.end_env()

    # ---------------------------------------------------------------------------- #
    #                           Environment functionality                          #
    # ---------------------------------------------------------------------------- #
    def get_schedulers(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Extract all schedulers available in the building model to be controlled.

        Args:
            path (str, optional): If path is specified, then this method export a xlsx file version in addition to return the dictionary.

        Returns:
            Dict[str, Any]: Python Dictionary: For each scheduler found, it shows type value and where this scheduler is present (Object name, Object field and Object type).
        """
        schedulers = self.simulator._config.schedulers
        if path is not None:
            export_actuators_to_excel(actuators=schedulers, path=path)
        return schedulers

    def get_zones(self) -> List[str]:
        """Get the zone names available in the building model of that environment.

        Returns:
            List[str]: List of the zone names.
        """
        return self.simulator._config.idf_zone_names

    def _get_action(self, action: Any) -> Union[int,
                                                float,
                                                np.integer,
                                                np.ndarray,
                                                List[Any],
                                                Tuple[Any]]:
        """Transform the action for sending it to the simulator."""

        # Get action depending on flag_discrete
        if self.flag_discrete:
            # Index for action_mapping
            if np.issubdtype(type(action), np.integer):
                if isinstance(action, int):
                    setpoints = self.action_mapping[action]
                else:
                    setpoints = self.action_mapping[action.item()]
            # Manual action
            elif isinstance(action, tuple) or isinstance(action, list):
                # stable-baselines DQN bug prevention
                if len(action) == 1:
                    setpoints = self.action_mapping[action[0]]
                else:
                    setpoints = action
            elif isinstance(action, np.ndarray):
                setpoints = self.action_mapping[action.item()]
            else:
                raise RuntimeError(
                    'action type not supported by Sinergym environment')
            action_ = list(setpoints)
        else:
            # transform action to setpoints simulation
            action_ = self._setpoints_transform(action)

        return action_

    def _setpoints_transform(self,
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
            if self._action_space.low[i] <= value <= self._action_space.high[i]:
                a_max_min = self._action_space.high[i] - \
                    self._action_space.low[i]
                sp_max_min = self.setpoints_space.high[i] - \
                    self.setpoints_space.low[i]

                action_.append(
                    self.setpoints_space.low[i] +
                    (
                        value -
                        self._action_space.low[i]) *
                    sp_max_min /
                    a_max_min)
            else:
                # If action is outer action_space already, it don't need
                # transformation
                action_.append(value)

        return action_

    def _check_eplus_env(self) -> None:
        """This method checks that environment definition is correct and it has not inconsistencies.
        """
        # OBSERVATION
        assert len(self.variables['observation']) == self._observation_space.shape[
            0], 'Observation space has not the same length than variable names specified.'

        # ACTION
        if self.flag_discrete:
            assert hasattr(
                self, 'action_mapping'), 'Discrete environment: action mapping should have been defined.'
            assert not hasattr(
                self, 'setpoints_space'), 'Discrete environment: setpoints space should not have been defined.'
            assert self._action_space.n == len(
                self.action_mapping), 'Discrete environment: The length of the action_mapping must match the dimension of the discrete action space.'
            for values in self.action_mapping.values():
                assert len(values) == len(
                    self.variables['action']), 'Discrete environment: Action mapping tuples values must have the same length than action variables specified.'
        else:
            assert len(self.variables['action']) == self._action_space.shape[
                0], 'Action space shape must match with number of action variables specified.'
            assert hasattr(
                self, 'setpoints_space'), 'Continuous environment: setpoints_space attribute should have been defined.'
            assert not hasattr(
                self, 'action_mapping'), 'Continuous environment: action mapping should not have been defined.'
            assert len(self._action_space.low) == len(self.variables['action']) and len(self._action_space.high) == len(
                self.variables['action']), 'Continuous environment: low and high values action space definition should have the same number of values than action variables.'

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
