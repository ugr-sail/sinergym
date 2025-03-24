"""
Class for connecting EnergyPlus with Python using pyenergyplus API.
"""

import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

from pyenergyplus.api import EnergyPlusAPI
from tqdm import tqdm

from sinergym.utils.common import *
from sinergym.utils.constants import LOG_SIM_LEVEL
from sinergym.utils.logger import TerminalLogger


class EnergyPlus(object):

    # ---------------------------------------------------------------------------- #
    #                           Simulator Terminal Logger                          #
    # ---------------------------------------------------------------------------- #

    logger = TerminalLogger().getLogger(
        name='SIMULATOR',
        level=LOG_SIM_LEVEL)

    def __init__(
            self,
            name: str,
            obs_queue: Queue,
            info_queue: Queue,
            act_queue: Queue,
            context_queue: Queue,
            time_variables: List[str] = [],
            variables: Dict[str, Tuple[str, str]] = {},
            meters: Dict[str, str] = {},
            actuators: Dict[str, Tuple[str, str, str]] = {},
            context: Dict[str, Tuple[str, str, str]] = {}):
        """EnergyPlus runner class. This class run an episode in a thread when start() is called.

        Args:
            name (str): Name of the environment which is using the simulator.
            obs_queue (Queue): Observation queue for Gymnasium environment communication.
            info_queue (Queue): Extra information dict queue for Gymnasium environment communication.
            act_queue (Queue): Action queue for Gymnasium environment communication.
            context_queue (Queue): Context queue for Gymnasium environment communication, modifying internal states.
            time_variables (List[str]): EnergyPlus time variables we want to observe. The name of the variable must match with the name of the E+ Data Transfer API method name. Defaults to empty list
            variables (Dict[str, Tuple[str, str]]): Specification for EnergyPlus Output:Variable. The key name is custom, then tuple must be the original variable name and the output variable key. Defaults to empty dict.
            meters (Dict[str, str]): Specification for EnergyPlus Output:Meter. The key name is custom, then value is the original EnergyPlus Meters name.
            actuators (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Input Actuators. The key name is custom, then value is a tuple with actuator type, value type and original actuator name. Defaults to empty dict.
            context (Dict[str, Tuple[str, str, str]]): Specification for EnergyPlus Context Actuators. The key name is custom, then value is a tuple with actuator type, value type and original actuator name. These values are processed as real-time building configuration instead of real-time control. Defaults to empty dict.
        """

        # ---------------------------------------------------------------------------- #
        #                               Attributes set up                              #
        # ---------------------------------------------------------------------------- #
        self.name = name
        # ------------------------- Gym communication queues ------------------------- #
        self.obs_queue = obs_queue
        self.info_queue = info_queue
        self.act_queue = act_queue
        self.context_queue = context_queue

        # ------------------------------ Warmup process ------------------------------ #
        self.warmup_queue = Queue()
        self.warmup_complete = False

        # -------------------------------- API Objects ------------------------------- #
        self.api = EnergyPlusAPI()
        self.exchange = self.api.exchange

        # --------------------------------- Handlers --------------------------------- #
        self.var_handlers: Optional[Dict[str, int]] = None
        self.meter_handlers: Optional[Dict[str, int]] = None
        self.actuator_handlers: Optional[Dict[str, int]] = None
        self.context_handlers: Optional[Dict[str, int]] = None
        self.available_data: Optional[str] = None

        # --------------------- Simulation elements to read/write -------------------- #
        self.time_variables = time_variables
        self.variables = variables
        self.meters = meters
        self.actuators = actuators
        self.context = context

        # ----------------------------- Simulation thread ---------------------------- #
        self.energyplus_thread: Optional[threading.Thread] = None
        self.energyplus_state: Optional[int] = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized_handlers = False
        self.system_ready = False
        self.simulation_complete = False

        # ----------------------------------- Paths ---------------------------------- #
        self._building_path: Optional[str] = None
        self._weather_path: Optional[str] = None
        self._output_path: Optional[str] = None

        self.logger.debug('Energyplus simulator initialized.')

    # ---------------------------------------------------------------------------- #
    #                                 Main methods                                 #
    # ---------------------------------------------------------------------------- #
    def start(self,
              building_path: str,
              weather_path: str,
              output_path: str,
              episode: int) -> None:
        """Initializes all callbacks and handlers using EnergyPlus API, prepare the simulation system
           and start running the simulation in a Python thread.

        Args:
            building_path (str): EnergyPlus input description file path.
            weather_path (str): EnergyPlus weather path.
            output_path (str): Path where EnergyPlus process is going to allocate its output files.
            episode (int): Number of the episode to run (useful to show in progress bar).
        """

        # ------------------------------ Path attributes ----------------------------- #
        self._building_path = building_path
        self._weather_path = weather_path
        self._output_path = output_path

        # ------------------------- Initiate Energyplus state ------------------------ #
        self.energyplus_state = self.api.state_manager.new_state()

        # --------------------- Disable default Energyplus Output -------------------- #
        self.api.runtime.set_console_output_status(
            self.energyplus_state, False)

        # ------------------------ Progress bar for simulation ----------------------- #
        self.progress_bar = None

        # ------------------------- Callback for progress bar ------------------------ #
        def _progress_update(percent: int) -> None:
            if self.system_ready:

                if self.progress_bar is None:
                    # Progress bar for simulation
                    self.progress_bar = tqdm(
                        total=100,
                        desc=f'Simulation Progress [Episode {episode}]',
                        ncols=100,
                        unit='%',
                        leave=True,
                        position=0,
                        ascii=False,
                        dynamic_ncols=True,
                        file=sys.stdout)

                percent = percent + 1 if percent < 100 else percent
                self.progress_bar.update(percent - self.progress_bar.n)
                self.progress_bar.set_postfix_str(f'{percent}% completed')
                self.progress_bar.refresh()

        # ------------------------- Main Callbacks definition ------------------------ #
        self.api.runtime.callback_progress(
            self.energyplus_state, _progress_update)

        self.api.runtime.callback_after_new_environment_warmup_complete(
            self.energyplus_state, self._warmup_complete)
        # Warmup --> init system --> collect obs, action and context

        # ------------------- Run EnergyPlus in a non-blocking way ------------------- #
        def _run_energyplus(runtime, cmd_args, state, results):
            self.logger.debug(
                f'Running EnergyPlus with args: {cmd_args}')

            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)
            self.simulation_complete = True

        # ------------------ Creating the thread and start execution ----------------- #
        self.energyplus_thread = threading.Thread(
            target=_run_energyplus,
            name=self.name,
            args=(
                self.api.runtime,
                self.make_eplus_args(),
                self.energyplus_state,
                self.sim_results
            ),
            daemon=True
        )

        self.energyplus_thread.start()
        self.logger.debug('Energyplus thread started.')

    def stop(self) -> None:
        """It forces the simulation ends, cleans all communication queues, thread is deleted (joined) and simulator attributes are
           reset (except handlers, to not initialize again if there is a next thread execution).
        """
        if self.is_running:
            # Kill progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()
            # Set simulation as complete and force thread to finish
            self.simulation_complete = True
            # Unblock action thread if needed
            if self.act_queue.empty():
                self.act_queue.put([0] * len(self.actuators))
            # Wait to thread to finish (without control)
            self.energyplus_thread.join()
            self._flush_queues()
            # Delete thread
            self.energyplus_thread = None
            # Clean runtime callbacks
            self.api.runtime.clear_callbacks()
            # Clean Energyplus state
            self.api.state_manager.delete_state(
                self.energyplus_state)
            # Set flags to default value
            self.sim_results: Dict[str, Any] = {}
            self.warmup_complete = False
            self.initialized_handlers = False
            self.system_ready = False
            self.simulation_complete = False

            self.logger.debug('Energyplus thread stopped.')

    def failed(self) -> bool:
        """Method to determine if simulation has failed.

        Returns:
            bool: Flag to describe this state
        """
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """Transform attributes defined in class instance into energyplus bash command

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
    #                             E+ Callbacks methods                             #
    # ---------------------------------------------------------------------------- #

    # Warmup --> init system --> collect obs, action and context

    # ------------------------------ Warmup Process ------------------------------ #

    def _warmup_complete(self, state_argument: Any) -> None:
        """Callback that indicates when warmup process has been completed successfully. It sets the warmup flag to True and
           put a flag in the warmup queue to indicate that warmup process has been completed. Then, it initializes the system.

        Args:
            state_argument (int): EnergyPlus API state
        """
        if not self.exchange.warmup_flag(
                state_argument) and not self.warmup_complete:
            self.warmup_complete = True
            self.warmup_queue.put(True)
            self.logger.info(
                'Warmup process has been completed successfully.')

            # Init system after warmup
            self.api.runtime.callback_begin_new_environment(
                self.energyplus_state, self._init_system)

    # --------------------------- Initilization process -------------------------- #

    def _init_system(self, state_argument: int) -> None:
        """Indicate whether system are ready to work. After waiting to API data is available, handlers are initialized, and warmup flag is correct.

        Args:
            state_argument (int): EnergyPlus API state
        """
        if not self.system_ready:

            if self._init_handlers(state_argument):
                self.system_ready = True

                # register callback used to collect observations
                self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
                    self.energyplus_state, self._collect_obs_and_info)

                # register callback used to send actions
                self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
                    self.energyplus_state, self._process_action)

                # register callback used to process context
                self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
                    self.energyplus_state, self._process_context)

                self.logger.info('System is ready.')

    def _init_handlers(self, state_argument: int) -> bool:
        """initialize sensors/actuators handlers to interact with during simulation.

        Args:
            state_argument (int): EnergyPlus API state

        Returns:
            bool: Flag to indicate if handlers are initialized

        """
        # api data must be fully ready, else nothing happens
        if self.exchange.api_data_fully_ready(
                state_argument) and not self.initialized_handlers:

            if self.var_handlers is None and self.meter_handlers is None and self.actuator_handlers is None:
                # Get variable handlers using variables info
                self.var_handlers = {
                    key: self.exchange.get_variable_handle(state_argument, *var)
                    for key, var in self.variables.items()
                }

                # Get meter handlers using meters info
                self.meter_handlers = {
                    key: self.exchange.get_meter_handle(state_argument, meter)
                    for key, meter in self.meters.items()
                }

                # Get actuator handlers using actuators info
                self.actuator_handlers = {
                    key: self.exchange.get_actuator_handle(
                        state_argument, *actuator)
                    for key, actuator in self.actuators.items()
                }

                # Get context handlers using context info
                self.context_handlers = {
                    key: self.exchange.get_actuator_handle(
                        state_argument, *context)
                    for key, context in self.context.items()
                }

                # Save available_data information
                self.available_data = self.exchange.list_available_api_data_csv(
                    state_argument).decode('utf-8')

                # write available_data.csv in parent output_path
                parent_dir = Path(
                    self._output_path).parent.parent.absolute().__str__()
                data = self.available_data.splitlines()
                with open(parent_dir + '/data_available.txt', "w") as txt_file:
                    txt_file.writelines([line + '\n' for line in data])

                # Check handlers specified exists
                for variable_name, handle_value in self.var_handlers.items():
                    if handle_value < 0:
                        self.logger.error(
                            f'Variable handlers: {variable_name} is not an available variable, check your variable names and be sure that exists in <env-path>/data_available.txt')
                        # raise ValueError

                for meter_name, handle_value in self.meter_handlers.items():
                    if handle_value < 0:
                        self.logger.error(
                            f'Meter handlers: {meter_name} is not an available meter, check your meter names and be sure that exists in <env-path>/data_available.txt')
                        # raise ValueError

                for actuator_name, handle_value in self.actuator_handlers.items():
                    if handle_value < 0:
                        self.logger.error(
                            f'Actuator handlers: {actuator_name} is not an available actuator, check your actuator names and be sure that exists in <env-path>/data_available.txt')
                        # raise ValueError

            self.logger.info('handlers are ready.')
            self.initialized_handlers = True

        return self.initialized_handlers

    # ---------------- Observation, action and context processing ---------------- #

    def _collect_obs_and_info(self, state_argument: int) -> None:
        """EnergyPlus callback that collects output variables and info
        values and enqueue them in each simulation timestep.

        Args:
            state_argument (int): EnergyPlus API state
        """

        # if simulation is completed or not initialized --> do nothing
        if self.simulation_complete:
            self.api.runtime.stop_simulation(self.energyplus_state)

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
            # variables (getting value from handlers)
            ** {
                key: self.exchange.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handlers.items()
            },
            # meters (getting value from handlers)
            **{
                key: self.exchange.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handlers.items()
            }
        }

        # Mount the info dict in queue
        self.next_info = {
            # 'timestep': self.exchange.system_time_step(state_argument),
            'time_elapsed(hours)': self.exchange.current_sim_time(state_argument),
            'month': self.exchange.month(state_argument),
            'day': self.exchange.day_of_month(state_argument),
            'hour': self.exchange.hour(state_argument),
            'is_raining': self.exchange.is_raining(state_argument)
        }

        # Put in the queues the observation and info
        # self.logger.debug(f 'OBSERVATION put in QUEUE: {self.next_obs}')
        self.obs_queue.put(self.next_obs)
        # self.logger.debug(f'INFO put in QUEUE: {self.next_obs}')
        self.info_queue.put(self.next_info)

    def _process_action(self, state_argument: int) -> None:
        """EnergyPlus callback that sets output actuator value(s) from last received action.

        Args:
            state_argument (int): EnergyPlus API state
        """

        # If simulation is complete or not initialized --> do nothing
        if self.simulation_complete:
            self.api.runtime.stop_simulation(self.energyplus_state)

        # Get next action from queue and check type
        next_action = self.act_queue.get()
        # self.logger.debug(f'ACTION get from queue: {next_action}')
        if not self.simulation_complete:
            # Set the action values obtained in actuator handlers
            for i, (act_name, act_handle) in enumerate(
                    self.actuator_handlers.items()):
                self.exchange.set_actuator_value(
                    state=state_argument,
                    actuator_handle=act_handle,
                    actuator_value=next_action[i]
                )

                # self.logger.debug(
                #     f'Set in actuator {act_name} value {next_action[i]}.')

    def _process_context(self, state_argument: int) -> None:
        """EnergyPlus callback that sets actuator as a building context, instead of control.

        Args:
            state_argument (int): EnergyPlus API state
        """

        # If simulation is complete or not initialized --> do nothing
        if self.simulation_complete:
            self.api.runtime.stop_simulation(self.energyplus_state)

        # Get next action from queue and check type
        try:
            next_context = self.context_queue.get(block=False)
            if not self.simulation_complete:
                # Set the context values obtained in context handlers
                # (actuators)
                for i, (context_name, context_handle) in enumerate(
                        self.context_handlers.items()):
                    self.exchange.set_actuator_value(
                        state=state_argument,
                        actuator_handle=context_handle,
                        actuator_value=next_context[i]
                    )
        except Empty:
            pass

    # ---------------------------------------------------------------------------- #
    #                               Auxiliar methods                               #
    # ---------------------------------------------------------------------------- #

    def _flush_queues(self) -> None:
        """It empties all values allocated in observation, action and warmup queues
        """
        for q in [
                self.obs_queue,
                self.act_queue,
                self.info_queue,
                self.context_queue,
                self.warmup_queue]:
            while not q.empty():
                q.get()
        self.logger.debug('Simulator queues emptied.')

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #

    @property
    def is_running(self) -> bool:
        return self.energyplus_thread is not None
