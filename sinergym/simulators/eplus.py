"""
Class for connecting EnergyPlus with Python using Ptolomy server.

..  note::
    Modified from Zhiang Zhang's original project: https://github.com/zhangzhizza/Gym-Eplus
"""


import _thread
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from sinergym.utils.common import *
from sinergym.utils.config import Config

LOG_LEVEL_MAIN = 'INFO'
LOG_LEVEL_EPLS = 'FATAL'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"


class EnergyPlus(object):

    def __init__(
            self,
            eplus_path: str,
            weather_path: str,
            bcvtb_path: str,
            variable_path: str,
            idf_path: str,
            env_name: str,
            act_repeat: int = 1,
            max_ep_data_store_num: int = 10,
            config_params: Optional[Dict[str, Any]] = None):
        """EnergyPlus simulation class.

        Args:
            eplus_path (str):  EnergyPlus installation path.
            weather_path (str): EnergyPlus weather file (.epw) path.
            bcvtb_path (str): BCVTB installation path.
            variable_path (str): Path to variables file.
            idf_path (str): EnergyPlus input description file (.idf) path.
            env_name (str): The environment name.
            act_repeat (int, optional): The number of times to repeat the control action. Defaults to 1.
            max_ep_data_store_num (int, optional): The number of simulation results to keep. Defaults to 10.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """
        self._env_name = env_name
        self._thread_name = threading.current_thread().getName()
        self.logger_main = Logger().getLogger(
            'EPLUS_ENV_%s_%s_ROOT' %
            (env_name, self._thread_name), LOG_LEVEL_MAIN, LOG_FMT)

        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path
        # Create a socket for communication with the EnergyPlus
        self.logger_main.debug('Creating socket for communication...')
        self._socket = socket.socket()
        # Get local machine name
        self._host = socket.gethostname()
        # Bind to the host and any available port
        self._socket.bind((self._host, 0))
        # Get the port number
        sockname = self._socket.getsockname()
        self._port = sockname[1]
        # Listen on request
        self._socket.listen(60)

        self.logger_main.debug(
            'Socket is listening on host %s port %d' % (sockname))

        # Path attributes
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._variable_path = variable_path
        self._idf_path = idf_path
        # Episode existed
        self._episode_existed = False

        self._epi_num = 0
        self._act_repeat = act_repeat
        self._max_ep_data_store_num = max_ep_data_store_num
        self._last_action = [21.0, 25.0]

        # Creating models config (with extra params if exits)
        self._config = Config(
            idf_path=self._idf_path,
            weather_path=self._weather_path,
            env_name=self._env_name,
            max_ep_store=self._max_ep_data_store_num,
            extra_config=config_params)

        # Annotate experiment path in simulator
        self._env_working_dir_parent = self._config.experiment_path
        # Updating IDF file (Location and DesignDays) with EPW file
        self.logger_main.info(
            'Updating idf Site:Location and SizingPeriod:DesignDay(s) to weather and ddy file...')
        self._config.adapt_idf_to_epw()
        # Setting up extra configuration if exists
        self.logger_main.info(
            'Setting up extra configuration in building model if exists...')
        self._config.apply_extra_conf()
        # In this lines Epm model is modified but no IDF is stored anywhere yet

        # Eplus run info
        (self._eplus_run_st_mon,
         self._eplus_run_st_day,
         self._eplus_run_st_year,
         self._eplus_run_ed_mon,
         self._eplus_run_ed_day,
         self._eplus_run_ed_year,
         self._eplus_run_st_weekday,
         self._eplus_n_steps_per_hour) = self._config._get_eplus_run_info()

        # Eplus one epi len
        self._eplus_one_epi_len = self._config._get_one_epi_len()
        # Stepsize in seconds
        self._eplus_run_stepsize = 3600 / self._eplus_n_steps_per_hour

    def reset(
        self, weather_variability: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, List[float], bool]:
        """Resets the environment.
        This method does the following:
        1. Makes a new EnergyPlus working directory.
        2. Copies .idf and variables.cfg file to the working directory.
        3. Creates the socket.cfg file in the working directory.
        4. Creates the EnergyPlus subprocess.
        5. Establishes the socket connection with EnergyPlus.
        6. Reads the first sensor data from the EnergyPlus.
        7. Uses a new weather file if passed.

        Args:
            weather_variability (Optional[Tuple[float, float, float]], optional): Tuple with the sigma, mean and tau for OU process. Defaults to None.

        Returns:
            Tuple[float, List[float], bool]: The first element is a value with simulation time elapsed;
            the second element consist on EnergyPlus results in a 1-D list corresponding to the variables in
            variables.cfg and year, month, day and hour in simulation. The last element is a boolean indicating whether the episode terminates.
        """
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
            self.logger_main.info(
                'EnergyPlus episode completed successfully. ')
            self._epi_num += 1

        # Create EnergyPlus simulation process
        self.logger_main.info('Creating new EnergyPlus simulation episode...')
        # Creating episode working dir
        eplus_working_dir = self._config.set_episode_working_dir()
        # Getting IDF, WEATHER, VARIABLES and OUTPUT path for current episode
        eplus_working_idf_path = self._config.save_building_model()
        eplus_working_var_path = (eplus_working_dir + '/' + 'variables.cfg')
        eplus_working_out_path = (eplus_working_dir + '/' + 'output')
        eplus_working_weather_path = self._config.apply_weather_variability(
            variation=weather_variability)
        # Copy the variable.cfg file to the working dir
        copyfile(self._variable_path, eplus_working_var_path)

        self._create_socket_cfg(self._host,
                                self._port,
                                eplus_working_dir)
        # Create the socket.cfg file in the working dir
        self.logger_main.info('EnergyPlus working directory is in %s'
                              % (eplus_working_dir))
        # Create new random weather file in case variability was specified
        # noise always from original EPW

        # Select new weather if it is passed into the method
        eplus_process = self._create_eplus(
            self._eplus_path,
            eplus_working_weather_path,
            eplus_working_idf_path,
            eplus_working_out_path,
            eplus_working_dir)
        self.logger_main.debug(
            'EnergyPlus process is still running ? %r' %
            self._get_is_subprocess_running(eplus_process))
        self._eplus_process = eplus_process

        # Log EnergyPlus output
        eplus_logger = Logger().getLogger('EPLUS_ENV_%s_%s-EPLUSPROCESS_EPI_%d' %
                                          (self._env_name, self._thread_name, self._epi_num), LOG_LEVEL_EPLS, LOG_FMT)
        _thread.start_new_thread(self._log_subprocess_info,
                                 (eplus_process.stdout,
                                  eplus_logger))
        _thread.start_new_thread(self._log_subprocess_err,
                                 (eplus_process.stderr,
                                  eplus_logger))

        # Establish connection with EnergyPlus
        # Establish connection with client
        conn, addr = self._socket.accept()
        self.logger_main.debug('Got connection from %s at port %d.' % (addr))
        # Start the first data exchange
        rcv_1st = conn.recv(2048).decode(encoding='ISO-8859-1')
        self.logger_main.debug(
            'Got the first message successfully: ' + rcv_1st)
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
            = self._disassembleMsg(rcv_1st)
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, curSimTim)
        # Add time_info date in the end of the Energyplus observation
        Dblist = time_info + Dblist
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag]
        self._curSimTim = curSimTim
        # Check if episode terminates
        is_terminal = False
        if curSimTim >= self._eplus_one_epi_len:
            is_terminal = True
        # Change some attributes
        self._conn = conn
        self._eplus_working_dir = eplus_working_dir
        self._episode_existed = True
        # Check termination
        if is_terminal:
            self._end_episode()

        return (curSimTim, Dblist, is_terminal)

    def step(self, action: Union[int, float, np.integer, np.ndarray, List[Any],
                                 Tuple[Any]]
             ) -> Tuple[float, List[float], bool]:
        """Executes a given action.
        This method does the following:
        1. Sends a list of floats to EnergyPlus.
        2. Receives EnergyPlus results for the next step (state).

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Control actions that will be passed to EnergyPlus.

        Raises:
            RuntimeError: When you try to step in an terminated episode (you should be reset before).

        Returns:
            Tuple[float, List[float], bool]: The first element is a float with simulation time elapsed;
            the second element consist on EnergyPlus results in a 1-D list corresponding to the variables in
            variables.cfg. The last element is a boolean indicating whether the episode terminates.
        """
        # Check if terminal
        if self._curSimTim >= self._eplus_one_epi_len:
            raise RuntimeError(
                'You are trying to step in a terminated episode (do reset before).')
        # Send to EnergyPlus
        act_repeat_i = 0
        is_terminal = False
        curSimTim = self._curSimTim

        while act_repeat_i < self._act_repeat and (not is_terminal):
            self.logger_main.debug('Perform one step.')
            header = self._eplus_msg_header
            runFlag = 0  # 0 is normal flag
            tosend = self._assembleMsg(header[0], runFlag, len(action), 0,
                                       0, curSimTim, action)
            self._conn.send(tosend.encode())
            # Recieve from EnergyPlus
            rcv = self._conn.recv(2048).decode(encoding='ISO-8859-1')
            self.logger_main.debug('Got message successfully: ' + rcv)
            # Process received msg
            version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                = self._disassembleMsg(rcv)
            if curSimTim >= self._eplus_one_epi_len:
                is_terminal = True
                # Remember the last action
                self._last_action = action
            act_repeat_i += 1
        # Construct the return, which is the state observation of the last step
        # plus the integral item
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, curSimTim)
        # Add time_info to the observation (year,month,day and hour) at the
        # beggining
        Dblist = time_info + Dblist
        # Add terminal state
        # Change some attributes
        self._curSimTim = curSimTim
        self._last_action = action

        return (curSimTim, Dblist, is_terminal)

    def _create_eplus(
            self,
            eplus_path: str,
            weather_path: str,
            idf_path: str,
            out_path: str,
            eplus_working_dir: str) -> subprocess.Popen:
        """Creates the EnergyPlus process.

        Args:
            eplus_path (str): EnergyPlus path.
            weather_path (str): Weather file path (.epw).
            idf_path (str): Building model path (.idf).
            out_path (str): Output path.
            eplus_working_dir (str): EnergyPlus working directory.

        Returns:
            subprocess.Popen: EnergyPlus process.
        """

        eplus_process = subprocess.Popen(
            '%s -w %s -d %s %s' %
            (eplus_path +
             '/energyplus',
             weather_path,
             out_path,
             idf_path),
            shell=True,
            cwd=eplus_working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        return eplus_process

    def _create_socket_cfg(self, host: str, port: int, write_dir: str) -> None:
        """Creates the socket required by BCVTB

        Args:
            host (str): Socket host name.
            port (int): Socket port number.
            write_dir (str): The saving directory.
        """

        top = Element('BCVTB-client')
        ipc = SubElement(top, 'ipc')
        socket = SubElement(ipc, 'socket', {'port': str(port),
                                            'hostname': host, })
        xml_str = tostring(top, encoding='ISO-8859-1').decode()

        with open(write_dir + '/' + 'socket.cfg', 'w+') as socket_file:
            socket_file.write(xml_str)

    def _get_file_name(self, file_path: str) -> str:
        """get filename from a given path (last element from that path)

        Args:
            file_path (str): path where we want to extract filename

        Returns:
            str: filename result.
        """
        path_list = file_path.split('/')
        return path_list[-1]

    def _log_subprocess_info(self, out: Any, logger: logging.Logger) -> None:
        """Logger info message from subprocess

        Args:
            out (Any): stdout type
            logger (logging.Logger): Simulator Logger running currently
        """
        for line in iter(out.readline, b''):
            logger.info(line.decode())

    def _log_subprocess_err(self, out: Any, logger: logging.Logger) -> None:
        """Logger err message from subprocess

        Args:
            out (Any): stdout type
            logger (logging.Logger): Simulator Logger running currently
        """
        for line in iter(out.readline, b''):
            logger.error(line.decode())

    def _get_is_subprocess_running(
        self,
        subprocess: subprocess.Popen
    ) -> bool:
        """It indicates whether simulator is running as subprocess.Popen or not.

        Args:
            subprocess (subprocess.Popen): Simulator subprocess running

        Returns:
            bool: Flag which indicates subprocess is running or not.
        """
        if subprocess.poll() is None:
            return True
        else:
            return False

    def get_is_eplus_running(self) -> bool:
        """It indicates whether simulator is running as subprocess.Popen or not.

        Returns:
            bool: Flag which indicates subprocess is running or not.
        """
        if hasattr(self, '_eplus_process'):
            return self._get_is_subprocess_running(self._eplus_process)
        else:
            return False

    def end_env(self) -> None:
        """Method called after finishing using the environment in order to close it."""

        self._end_episode()
        # self._socket.shutdown(socket.SHUT_RDWR);
        self._socket.close()
        self.logger_main.info(
            'EnergyPlus simulation closed successfully. ')

    def end_episode(self) -> None:
        """It ends current simulator episode."""
        self._end_episode()

    def _end_episode(self) -> None:
        """This process terminates the current EnergyPlus subprocess.
        It is usually called by the reset() function before it resets the EnergyPlus environment.
        """
        if self._episode_existed:
            # Send final msg to EnergyPlus
            header = self._eplus_msg_header
            # Terminate flag is 1.0, specified by EnergyPlus
            flag = 1.0
            action = self._last_action
            action_size = len(self._last_action)
            tosend = self._assembleMsg(header[0], flag, action_size, 0,
                                       0, self._curSimTim, action)
            self.logger_main.debug('Send final msg to Eplus.')
            self._conn.send(tosend.encode())
            # Recieve the final msg from Eplus
            rcv = self._conn.recv(2048).decode(encoding='ISO-8859-1')
            self.logger_main.debug('Final msg from Eplus: %s', rcv)
            self._conn.send(tosend.encode())  # Send again, don't know why
            # Remove the connection
            self._conn.close()
            self._conn = None
            # Process the output
            # Sleep the thread so EnergyPlus has time to do the post processing
            time.sleep(1)

            # Kill subprocess
            os.killpg(self._eplus_process.pid, signal.SIGTERM)
            self._episode_existed = False

    def _run_eplus_outputProcessing(self) -> None:
        # If simulator has not been running with reset at least one time this
        # method is null.
        if hasattr(self, '_eplus_working_dir'):
            eplus_outputProcessing_process = subprocess.Popen(
                '%s' %
                (self._eplus_path +
                 '/PostProcess/ReadVarsESO'),
                shell=True,
                cwd=self._eplus_working_dir +
                '/output',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid)

    def _assembleMsg(
            self,
            version: int,
            flag: int,
            nDb: int,
            nIn: int,
            nBl: int,
            curSimTim: float,
            Dblist: Union[float, np.ndarray]) -> str:
        """Assembles the sent message to EnergyPlus based on the protocol.
        The message must be a blank space separated string set with the fields defined as arguments.

        Args:
            version (str): EnergyPlus version.
            flag (str):
            nDb (str):
            nIn (str):
            nBl (str):
            curSimTim (str): Current simulation time.
            Dblist (str): State of enviroment (depending on variables:output fixed in IDF file).

        Returns:
            str: All values concatenated in correct format to send to EnergyPlus subprocess.
        """

        ret = ''
        ret += '%d' % (version)
        ret += ' '
        ret += '%d' % (flag)
        ret += ' '
        ret += '%d' % (nDb)
        ret += ' '
        ret += '%d' % (nIn)
        ret += ' '
        ret += '%d' % (nBl)
        ret += ' '
        ret += '%20.15e' % (curSimTim)
        ret += ' '
        for i in range(len(Dblist)):
            ret += '%20.15e' % (Dblist[i])
            ret += ' '
        ret += '\n'
        return ret

    def _disassembleMsg(self,
                        rcv: str) -> Tuple[int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           float,
                                           List[float]]:
        """Disassembles the received message from EnergyPlus (convert raw str in Energyplus protocol).
        The message will be transformed to a Tuple with message field separated.

        Args:
            rcv (str): Raw information received from Sinergym

        Returns:
            Tuple[int, int, int, int, int, float, Tuple[float, ...]]: version, flag, nDb, nIn, nBl, curSimTim and DBList (simulation variables input) fields processed.
        """

        rcv = rcv.split(' ')
        version = int(rcv[0])
        flag = int(rcv[1])
        nDb = int(rcv[2])
        nIn = int(rcv[3])
        nBl = int(rcv[4])
        curSimTim = float(rcv[5])
        Dblist = []
        for i in range(6, len(rcv) - 1):
            Dblist.append(float(rcv[i]))

        return (version, flag, nDb, nIn, nBl, curSimTim, Dblist)

    @property
    def start_year(self) -> int:
        """Returns the EnergyPlus simulation year.

        Returns:
            int: Simulation year.
        """

        return self._config.start_year

    @property
    def start_mon(self) -> int:
        """Returns the EnergyPlus simulation start month.

        Returns:
            int: Simulation start month.
        """
        return self._eplus_run_st_mon

    @property
    def start_day(self) -> int:
        """Returns the EnergyPlus simulation start day of the month.

        Returns:
            int: Simulation start day of the month.
        """
        return self._eplus_run_st_day

    @property
    def start_weekday(self) -> int:
        """Returns the EnergyPlus simulation start weekday. From 0 (Monday) to 6 (Sunday).

        Returns:
            int: Simulation start weekday.
        """
        return self._eplus_run_st_weekday

    @property
    def env_name(self) -> str:
        """Returns the environment name.

        Returns:
            str: Environment name
        """
        return self._env_name
