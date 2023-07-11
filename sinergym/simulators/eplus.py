"""
Class for connecting EnergyPlus with Python using pyenergyplus API.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from pyenergyplus.api import EnergyPlusAPI

import numpy as np
from queue import Queue, Empty, Full
from sinergym.utils.common import *
from logging import Logger


LOG_LEVEL_MAIN = 'INFO'
LOG_LEVEL_EPLS = 'FATAL'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"

VALID_TIME_VARIABLES = []


class EnergyPlus(object):

    def __init__(
            self,
            building_path: str,
            weather_path: str,
            output_path: str,
            obs_queue: Queue,
            info_queue: Queue,
            act_queue: Queue,
            time_variables: List[str] = [],
            variables: Dict[str, Tuple[str, str]] = {},
            meters: Dict[str, Tuple[str, str]] = {},
            actuators: Dict[str, Tuple[str, str]] = {}):
        """EnergyPlus simulation run class. This class run an episode

        Args:
            building_path (str): EnergyPlus input description file path.
            weather_path (str): EnergyPlus weather path.
            output_path (str): Path where EnergyPlus process is going to allocate its output files.
            obs_queue (Queue): Observation queue for Gymnasium environment communication.
            info_queue (Queue): Extra information dict queue for Gymnasium environment communication.
            act_queue (Queue): Action queue for Gymnasium environment communication.
            time_variables (List[str]): Time variables which composes part of the observation, such as year, month, day or hour. Default empty list.
            variables (Dict[str,Tuple[str,str]]): Observation variables info. Default empty dict.
            meters (Dict[str,Tuple[str,str]]): Observation meters info. Default empty dict.
            actuators (Dict[str,Tuple[str,str]]): Action actuators info. Default empty dict.
        """

        # ---------------------------------------------------------------------------- #
        #                               Attributes set up                              #
        # ---------------------------------------------------------------------------- #

        # Gym communication queues
        self.obs_queue = obs_queue
        self.info_queue = info_queue
        self.act_queue = act_queue

        # Warmup process
        self.warmup_queue = Queue()
        self.warmup_complete = False

        # API Objects
        self.api = EnergyPlusAPI()
        self.exchange = self.api.exchange

        # Progress Bar
        self.progress_value: int = 0

        # Handles
        self.var_handles: Dict[str, int] = {}
        self.meter_handles: Dict[str, int] = {}
        self.actuator_handles: Dict[str, int] = {}

        # Simulation elements to read/write
        self.time_variables = time_variables
        self.variables = variables
        self.meters = meters
        self.actuators = actuators

        # Simulation thread
        self.energyplus_thread: Optional[threading.Thread] = None
        self.energyplus_state: Optional[int] = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized_handles = False
        self.callbacks_ready = False
        self.simulation_complete = False

        # Logger
        # self.logger = Logger().getLogger(
        #     'EPLUS_ENV_EPLUS_API', LOG_LEVEL_MAIN, LOG_FMT)

        # Path attributes
        self._building_path = building_path
        self._weather_path = weather_path
        self._output_path = output_path

    # ---------------------------------------------------------------------------- #
    #                                 Main methods                                 #
    # ---------------------------------------------------------------------------- #

    def start(self) -> None:
        """Initializes all callbacks and handles required in the simulation
           and start running the thread generated.
        """

        # Initiate Energyplus state
        self.energyplus_state = self.api.state_manager.new_state()
        runtime = self.api.runtime

        # Disable default Energyplus Output
        # self.api.runtime.set_console_output_status(
        #     self.energyplus_state, False)

        # Register callback used to track simulation progress
        def _progress_update(percent: int) -> None:
            filled_length = int(80 * (percent / 100.0))
            bar = "*" * filled_length + '-' * (80 - filled_length)
            print(f'\rProgress: |{bar}| {percent}%', end="\r")

        runtime.callback_progress(self.energyplus_state, _progress_update)

        # register callback used to signal warmup complete
        def _warmup_complete(state: Any) -> None:
            self.warmup_complete = True
            self.warmup_queue.put(True)
            print('Warmup process has been completed successfully.')

        runtime.callback_after_new_environment_warmup_complete(
            self.energyplus_state, _warmup_complete)

        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(
            self.energyplus_state, self._collect_obs)

        # register callback used to collect info with extra information
        runtime.callback_end_zone_timestep_after_zone_reporting(
            self.energyplus_state, self._collect_info)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(
            self.energyplus_state, self._process_action)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            self.logger.info(
                'Running EnergyPlus with args: {}'.format(cmd_args))

            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)
            self.simulation_complete = True
            print('')

        # Creating the thread and start execution
        self.energyplus_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.api.runtime,
                self.make_eplus_args(),
                self.energyplus_state,
                self.sim_results
            ),
            daemon=True
        )

    def stop(self) -> None:
        """It is called when simulation ends, setting up the simulation complete flag to True, cleaning all queues,
           thread is deleted, callbacks are cleaned and Energyplus state removed.
        """
        if self.energyplus_thread:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_thread.join()
            self.energyplus_thread = None
            self.api.runtime.clear_callbacks()
            self.api.state_manager.delete_state(
                self.energyplus_state)

    def failed(self) -> bool:
        """Method to determine if simulation has failed.

        Returns:
            bool: Flag to describe this state
        """
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """Transform attributes specified in simulator into energyplus bash command

        Returns:
            List[str]: List of the argument components for energyplus bash command
        """
        eplus_args = []
        eplus_args += ["-w",
                       self._weather_path,
                       "-d",
                       self._output_path,
                       self._building_path]
        return eplus_args

    # ---------------------------------------------------------------------------- #
    #                              Auxiliary methods                               #
    # ---------------------------------------------------------------------------- #

    def _collect_obs(self, state_argument: int) -> None:
        """EnergyPlus callback that collects output variables/meters
        values and enqueue them

        Args:
            state_argument (int): EnergyPlus API state
        """

        # if simulation is completed or not initialized --> do nothing
        if self.simulation_complete:
            return
        if not self._init_callback(state_argument):
            print('PIPOPIPOPIPOI')
            return

        # Obtain observation (time_variables, variables and meters) values in dict
        # format
        self.next_obs = {
            # time variables (calling in exchange module directly)
            **{
                t_variable: eval('self.exchange.' +
                                 t_variable +
                                 '(self.energyplus_state)', {'self': self})
                for t_variable in self.time_variables
            },
            # variables (getting value from handles)
            ** {
                key: self.exchange.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handles.items()
            },
            # meters (getting value from handles)
            **{
                key: self.exchange.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handles.items()
            }
        }

        # Put in the queue the observation
        self.obs_queue.put(self.next_obs)

    def _collect_info(self, state_argument: int) -> None:
        """EnergyPlus callback that collects output info
        values and enqueue them

        Args:
            state_argument (int): EnergyPlus API state
        """

        # If simulation is complete or not initialized --> do nothing
        if self.simulation_complete:
            return
        if not self._init_callback(state_argument):
            print('PIPOPIPOPIPOI')
            return

        # Mount the info dict in queue
        self.next_info = {
            # 'timestep': self.exchange.system_time_step(state_argument),
            'time_elapsed': self.exchange.current_sim_time(state_argument),
            'year': self.exchange.year(state_argument),
            'month': self.exchange.month(state_argument),
            'day': self.exchange.day_of_month(state_argument),
            'hour': self.exchange.hour(state_argument),
            'is_raining': self.exchange.is_raining(state_argument)
        }

        self.info_queue.put(self.next_info)

    def _process_action(self, state_argument: int) -> None:
        """EnergyPlus callback that sets actuator value from last decided action

        Args:
            state_argument (int): EnergyPlus API state
        """

        # If simulation is complete or not initialized --> do nothing
        if self.simulation_complete:
            return
        if not self._init_callback(state_argument):
            return
        # If not value in action queue --> do nothing
        if self.act_queue.empty():
            return
        # Get next action from queue and check type
        next_action = self.act_queue.get()

        # Set the action values obtained in actuator handles
        for i, act_handle in enumerate(list(self.actuator_handles.values())):
            self.exchange.set_actuator_value(
                state=state_argument,
                actuator_handle=act_handle,
                actuator_value=next_action[i]
            )

    def _init_callback(self, state_argument: int) -> bool:
        """Indicate whether callbacks are ready to work.

        Args:
            state_argument (int): EnergyPlus API state

        Returns:
            bool: Flag to define whether handles and simulation is ready.
        """
        if not self.initialized_handles:
            self._init_handles(state_argument)
        self.callbacks_ready = self.initialized_handles and not self.exchange.warmup_flag(
            state_argument)
        return self.callbacks_ready

    def _init_handles(self, state_argument: int) -> bool:
        """initialize sensors/actuators handles to interact with during simulation.

        Args:
            state_argument (int): EnergyPlus API state

        """
        # api data must be fully ready, else nothing happens
        if self.exchange.api_data_fully_ready(state_argument):

            # Get variable handles using variables info
            self.var_handles = {
                key: self.exchange.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            # Get meter handles using meters info
            self.meter_handles = {
                key: self.exchange.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            # Get actuator handles using actuators info
            self.actuator_handles = {
                key: self.exchange.get_actuator_handle(
                    state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                self.var_handles,
                self.meter_handles,
                self.actuator_handles
            ]:
                # If any handle have value -1, it notifies error name
                if any([v == -1 for v in handles.values()]):
                    available_data = self.exchange.list_available_api_data_csv(
                        state_argument).decode('utf-8')
                    print(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}")
                    exit(1)

            self.initialized_handles = True

    def _flush_queues(self) -> None:
        """It empties all values allocated in observation, action and warmup queues
        """
        for q in [self.obs_queue, self.act_queue, self.warmup_queue]:
            while not q.empty():
                q.get()

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #

    # @property
    # def var_handles(self) -> Optional[int]:
    #     return self.var_handles

    # @property
    # def meter_handles(self) -> Optional[int]:
    #     return self.meter_handles

    # @property
    # def actuator_handles(self) -> Optional[int]:
    #     return self.actuator_handles
