"""
Class for connecting EnergyPlus with Python using Ptolomy server
Author: Modified from Zhiang Zhang https://github.com/zhangzhizza/Gym-Eplus
"""


import socket              
import os
import time
import copy
import signal
import _thread
import logging
import subprocess
import threading
import numpy as np
from shutil import copyfile, rmtree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

from ..utils.common import *


YEAR = 1991 # Non leap year
WEEKDAY_ENCODING = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}

CWD = os.getcwd();
LOG_LEVEL_MAIN = 'INFO';
LOG_LEVEL_EPLS = 'ERROR';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";


class EnergyPlus(object):
    """EnergyPlus simulation class.

    Args:
    eplus_path: String
      EnergyPlus executive command path.
    weather_path: String
      EnergyPlus weather file path (.epw). 
    bcvtb_path: String
      BCVTB installation path.
    variable_path: String
      variable.cfg path.
    idf_path: String
      EnergyPlus input description file (.idf) path.
    env_name: str
      The environment name.
    act_repeat: int
      The number of times to repeat the control action.
    max_ep_data_store_num: int
      The number of simulation results to keep.
    """

    def __init__(self, eplus_path, weather_path, bcvtb_path, variable_path, idf_path, env_name, 
                    act_repeat = 1, max_ep_data_store_num = 10):
        self._env_name = env_name;
        self._thread_name = threading.current_thread().getName();
        self.logger_main = Logger().getLogger('EPLUS_ENV_%s_%s_ROOT'%(env_name, self._thread_name), 
                                            LOG_LEVEL_MAIN, LOG_FMT);
        
        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path;
        # Create a socket for communication with the EnergyPlus
        self.logger_main.debug('Creating socket for communication...')
        s = socket.socket()
        host = socket.gethostname() # Get local machine name
        s.bind((host, 0))           # Bind to the host and any available port
        sockname = s.getsockname();
        port = sockname[1];         # Get the port number
        s.listen(60)                # Listen on request
        self.logger_main.debug('Socket is listening on host %s port %d'%(sockname));
        self._env_working_dir_parent = self._get_eplus_working_folder(CWD, '-%s-res'%(env_name));
        os.makedirs(self._env_working_dir_parent);
        self._host = host;
        self._port = port;
        self._socket = s;
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._variable_path = variable_path
        self._idf_path = idf_path
        self._episode_existed = False;
        (self._eplus_run_st_mon, self._eplus_run_st_day,
         self._eplus_run_ed_mon, self._eplus_run_ed_day,
         self._eplus_run_st_weekday,
         self._eplus_run_stepsize) = self._get_eplus_run_info(idf_path);
        self._eplus_run_stepsize = 3600 / self._eplus_run_stepsize # Stepsize in second
        self._eplus_one_epi_len = self._get_one_epi_len(self._eplus_run_st_mon,
                                                        self._eplus_run_st_day,
                                                        self._eplus_run_ed_mon,
                                                        self._eplus_run_ed_day);
        self._epi_num = 0;
        self._act_repeat = act_repeat;
        self._max_ep_data_store_num = max_ep_data_store_num;
        self._last_action = [21.0, 25.0]; 

    def reset(self, new_weather: str = None):
        """Reset the environment.

        This method does the followings:
        1: Make a new EnergyPlus working directory
        2: Copy .idf and variables.cfg file to the working directory
        3: Create the socket.cfg file in the working directory
        4: Create the EnergyPlus subprocess
        5: Establish the socket connection with EnergyPlus
        6: Read the first sensor data from the EnergyPlus
        7: Use a new weather file if passed
        
        Return: (float, [float], boolean)
            The index 0 is current_simulation_time in second, 
            index 1 is EnergyPlus results in 1-D python list requested by the 
            variable.cfg, index 2 is the boolean indicating whether the episode terminates.
        """
        ret = [];
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
            self.logger_main.info('Last EnergyPlus process has been closed. ')
            self._epi_num += 1;
        
        # Create EnergyPlus simulaton process
        self.logger_main.info('Creating EnergyPlus simulation environment...')
        eplus_working_dir = self._get_eplus_working_folder(
                                self._env_working_dir_parent, '-sub_run');
        os.makedirs(eplus_working_dir); # Create the Eplus working directory
        self._rm_past_history_dir(eplus_working_dir, '-sub_run'); # Remove redundant past working directories
        eplus_working_idf_path = (eplus_working_dir + 
                                  '/' + 
                                  self._get_file_name(self._idf_path));
        eplus_working_var_path = (eplus_working_dir + 
                                  '/' + 
                                  'variables.cfg'); # Variable file must be with this name
        eplus_working_out_path = (eplus_working_dir + 
                                  '/' + 
                                  'output');
        copyfile(self._idf_path, eplus_working_idf_path);
                                    # Copy the idf file to the working directory
        copyfile(self._variable_path, eplus_working_var_path);
                                    # Copy the variable.cfg file to the working dir
        self._create_socket_cfg(self._host, 
                                self._port,
                                eplus_working_dir); 
                                    # Create the socket.cfg file in the working dir
        self.logger_main.info('EnergyPlus working directory is in %s'
                              %(eplus_working_dir));
        # Select new weather if it is passed into the method
        weather_path = self._weather_path if new_weather is None else new_weather
        eplus_process = self._create_eplus(self._eplus_path, weather_path, 
                                            eplus_working_idf_path,
                                            eplus_working_out_path,
                                            eplus_working_dir);
        self.logger_main.debug('EnergyPlus process is still running ? %r' 
                                %self._get_is_subprocess_running(eplus_process))
        self._eplus_process = eplus_process;
        # Log the Eplus output
        eplus_logger = Logger().getLogger('EPLUS_ENV_%s_%s-EPLUSPROCESS_EPI_%d'
                                        %(self._env_name, self._thread_name, self._epi_num),
                                        LOG_LEVEL_EPLS, LOG_FMT);
        _thread.start_new_thread(self._log_subprocess_info,
                                (eplus_process.stdout,
                                 eplus_logger));
        _thread.start_new_thread(self._log_subprocess_err,
                                (eplus_process.stderr,
                                 eplus_logger));                              
        # Establish connection with EnergyPlus
        conn, addr = self._socket.accept()     # Establish connection with client.
        self.logger_main.debug('Got connection from %s at port %d.'%(addr));
        # Start the first data exchange
        rcv_1st = conn.recv(2048).decode(encoding = 'ISO-8859-1');
        self.logger_main.debug('Got the first message successfully: ' + rcv_1st);
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                                                = self._disassembleMsg(rcv_1st);
        ret.append(curSimTim);
        ret.append(Dblist);
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag];
        self._curSimTim = curSimTim;
        # Check if episode terminates
        is_terminal = False;
        if curSimTim >= self._eplus_one_epi_len:
            is_terminal = True;
        ret.append(is_terminal);
        # Change some attributes
        self._conn = conn;
        self._eplus_working_dir = eplus_working_dir;
        self._episode_existed = True;
        # Process for episode terminal
        if is_terminal:
            self._end_episode();
            
        return tuple(ret);

    def step(self, action):
        """Execute the specified action.
        
        This method does the followings:
        1: Send a list of float to EnergyPlus
        2: Recieve EnergyPlus results for the next step (state)

        Parameters
        ----------
        action: python list of float
          Control actions that will be passed to the EnergyPlus

        Return: (float, [float], boolean)
            The index 0 is current_simulation_time in second, 
            index 1 is EnergyPlus results in 1-D python list requested by the 
            variable.cfg, index 2 is the boolean indicating whether the episode terminates.
        """
        # Check terminal
        if self._curSimTim >= self._eplus_one_epi_len:
            return None;
        ret = [];
        # Send to the EnergyPlus
        act_repeat_i = 0;
        is_terminal = False;
        curSimTim = self._curSimTim;
        while act_repeat_i < self._act_repeat and (not is_terminal):
            self.logger_main.debug('Perform one step.')
            header = self._eplus_msg_header;
            runFlag = 0 # 0 is normal flag
            tosend = self._assembleMsg(header[0], runFlag, len(action), 0,
                                       0, curSimTim, action);
            self._conn.send(tosend.encode());
            # Recieve from the EnergyPlus
            rcv = self._conn.recv(2048).decode(encoding = 'ISO-8859-1');
            self.logger_main.debug('Got message successfully: ' + rcv);
            # Process received msg        
            version, flag, nDb, nIn, nBl, curSimTim, Dblist \
                                        = self._disassembleMsg(rcv);
            if curSimTim >= self._eplus_one_epi_len:
                is_terminal = True;
                # Remember the last action
                self._last_action = action;
            act_repeat_i += 1;
        # Construct the return. The return is the state observation of the last step plus the integral item
        ret.append(curSimTim);
        ret.append(Dblist);
        # Add terminal status
        ret.append(is_terminal);
        # Change some attributes
        self._curSimTim = curSimTim;
        self._last_action = action;
        return ret;
    
    def _rm_past_history_dir(self, cur_eplus_working_dir, dir_sig):
        """Remove the past simulation results.

        Args:
            cur_eplus_working_dir: str
                The current eplus working directory
            dir_sig: str
                The directory split signature

        Reture: None
        """

        cur_dir_name, cur_dir_id = cur_eplus_working_dir.split(dir_sig)
        cur_dir_id = int(cur_dir_id);
        if cur_dir_id - self._max_ep_data_store_num > 0:
            rm_dir_id = cur_dir_id - self._max_ep_data_store_num;
            rm_dir_full_name = cur_dir_name + dir_sig + str(rm_dir_id);
            rmtree(rm_dir_full_name);

    def _create_eplus(self, eplus_path, weather_path, 
                      idf_path, out_path, eplus_working_dir):
        
        eplus_process = subprocess.Popen('%s -w %s -d %s %s'
                        %(eplus_path + '/energyplus', weather_path, 
                          out_path, idf_path),
                        shell = True,
                        cwd = eplus_working_dir,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        preexec_fn=os.setsid);
        return eplus_process;
    
    def _get_eplus_working_folder(self, parent_dir, dir_sig = '-run'):
        """Return Eplus output folder. 

        Assumes folders in the parent_dir have suffix -run{run
        number}. Finds the highest run number and sets the output folder
        to that number + 1. 

        Parameters
        ----------
        parent_dir: str
        Parent dir of the Eplus output directory.

        Returns
        -------
        parent_dir/run_dir
        Path to Eplus save directory.
        """
        os.makedirs(parent_dir, exist_ok=True)
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split(dir_sig)[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, 'Eplus-env')
        parent_dir = parent_dir + '%s%d'%(dir_sig, experiment_id)
        return parent_dir

    def _create_socket_cfg(self, host, port, write_dir):
        """
        Create the socket.cfg required by BCVTB.

        Args:
            host: String
                Socket host name.
            port: int
                Socket port number.
            write_dir: String
                The write directory.
        Return: None
        """
        top = Element('BCVTB-client');
        ipc = SubElement(top, 'ipc');
        socket = SubElement(ipc, 'socket'
                            ,{'port':str(port),
                              'hostname':host,})
        xml_str = tostring(top, encoding='ISO-8859-1').decode();
        
        with open(write_dir + '/' + 'socket.cfg', 'w+') as socket_file:
            socket_file.write(xml_str);
    
    def _get_file_name(self, file_path):
        path_list = file_path.split('/');
        return path_list[-1];
    
    def _log_subprocess_info(self, out, logger):
        for line in iter(out.readline, b''):
            logger.info(line.decode())
            
    def _log_subprocess_err(self, out, logger):
        for line in iter(out.readline, b''):
            logger.error(line.decode())
            
    def _get_is_subprocess_running(self, subprocess):
        if subprocess.poll() is None:
            return True;
        else:
            return False;
        
    def get_is_eplus_running(self):
        return self._get_is_subprocess_running(self._eplus_process);
    
    def end_env(self):
        """
        This method is called after finishing using the environment. 
        """
        self._end_episode();
        #self._socket.shutdown(socket.SHUT_RDWR);
        self._socket.close();        
        
    def end_episode(self):
        self._end_episode();
            
    def _end_episode(self):
        """
        This method terminates the current EnergyPlus subprocess.
        
        This method is usually called by the reset() function before it
        resets the EnergyPlus environment.
        """
        # Send the final msg to EnergyPlus
        header = self._eplus_msg_header;
        flag = 1.0; # Terminate flag is 1.0, specified by EnergyPlus
        action = self._last_action;
        action_size = len(self._last_action);
        tosend = self._assembleMsg(header[0], flag, action_size, 0,
                                    0, self._curSimTim, action);
        self.logger_main.debug('Send the final msg to Eplus.');
        self._conn.send(tosend.encode());
        # Recieve the final msg from Eplus
        rcv = self._conn.recv(2048).decode(encoding = 'ISO-8859-1');
        self.logger_main.debug('Final msh from Eplus: %s', rcv)
        self._conn.send(tosend.encode()); # Send again, don't know why
        # Remove the connection
        self._conn.close();
        self._conn = None;
        # Process the output
        time.sleep(1);# Sleep the thread so EnergyPlus has time to do the
                      # post processing

        # Kill subprocess
        os.killpg(self._eplus_process.pid, signal.SIGTERM);
        self._episode_existed = False;
        
    def _run_eplus_outputProcessing(self):
        eplus_outputProcessing_process =\
         subprocess.Popen('%s'
                        %(self._eplus_path + '/PostProcess/ReadVarsESO'),
                        shell = True,
                        cwd = self._eplus_working_dir + '/output',
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        preexec_fn=os.setsid)
         
        
    def _assembleMsg(self, version, flag, nDb, nIn, nBl, curSimTim, Dblist):
        """
        Assemble the send msg to the EnergyPlus based on the protocal.
        Send msg must a blank space seperated string, [verison, flag, nDb
        , nIn, nBl, curSimTim, float, float, float ....]
        
        Return:
            The send msg.
        """
        ret = '';
        ret += '%d'%(version);
        ret += ' ';
        ret += '%d'%(flag);
        ret += ' ';
        ret += '%d'%(nDb);
        ret += ' ';
        ret += '%d'%(nIn);
        ret += ' ';
        ret += '%d'%(nBl);
        ret += ' ';
        ret += '%20.15e'%(curSimTim);
        ret += ' ';
        for i in range(len(Dblist)):
            ret += '%20.15e'%(Dblist[i]);
            ret += ' ';
        ret += '\n';
        
        return ret;
    
    def _disassembleMsg(self, rcv):
        rcv = rcv.split(' ');
        version = int(rcv[0]);
        flag = int(rcv[1]);
        nDb = int(rcv[2]);
        nIn = int(rcv[3]);
        nBl = int(rcv[4]);
        curSimTim = float(rcv[5]);
        Dblist = [];
        for i in range(6, len(rcv) - 1):
            Dblist.append(float(rcv[i]));
        
        return (version, flag, nDb, nIn, nBl, curSimTim, Dblist);
    
    def _get_eplus_run_info(self, idf_path):
        """
        This method read the .idf file and find the running start month, start
        date, end month, end date and the step size.
        
        Args:
            idf_path: String
                The .idf file path.
        
        Return: (int, int, int, int, int, int)
            (start month, start date, end month, end date, start weekday, 
            step size)
        """
        ret = [];
        
        with open(idf_path, encoding = 'ISO-8859-1') as idf:
            contents = idf.readlines();
        
        # Run period
        tgtIndex = None;
        
        for i in range(len(contents)):
            line = contents[i];
            effectiveContent = line.strip().split('!')[0] # Ignore contents after '!'
            effectiveContent = effectiveContent.strip().split(',')[0]
                                                          # Remove tailing ','
            if effectiveContent.lower() == 'runperiod':
                tgtIndex = i;
                break;
        
        for i in range(2, 6):
            ret.append(int(contents[tgtIndex + i].strip()
                                                 .split('!')[0]
                                                 .strip()
                                                 .split(',')[0]
                                                 .strip()
                                                 .split(';')[0]));
        # Start weekday
        ret.append(WEEKDAY_ENCODING[contents[tgtIndex + i + 1].strip()
                                                          .split('!')[0]
                                                          .strip()
                                                          .split(',')[0]
                                                          .strip()
                                                          .split(';')[0]
                                                          .strip()
                                                          .lower()]);
        # Step size
        line_count = 0;
        for line in contents:
            effectiveContent = line.strip().split('!')[0] # Ignore contents after '!'
            effectiveContent = effectiveContent.strip().split(',');
            if effectiveContent[0].strip().lower() == 'timestep':
                if len(effectiveContent) > 1 and len(effectiveContent[1]) > 0:
                    ret.append(int(effectiveContent[1]
                                   .split(';')[0]
                                   .strip()));
                else:
                    ret.append(int(contents[line_count + 1].strip()
                                                  .split('!')[0]
                                                  .strip()
                                                  .split(',')[0]
                                                  .strip()
                                                  .split(';')[0]));
                break;
            line_count += 1;
            
        return tuple(ret);
            
    def _get_one_epi_len(self, st_mon, st_day, ed_mon, ed_day):
        """
        Get the length of one episode (One EnergyPlus process run to the end).
        
        Args:
            st_mon, st_day, ed_mon, ed_day: int
                The EnergyPlus simulation start month, start day, end month, 
                end day.
        
        Return: int
            The simulation time step that the simulation ends.
        """
        return get_delta_seconds(YEAR, st_mon, st_day, ed_mon, ed_day);
    
    @property
    def start_year(self):
        """
        Return the EnergyPlus simulaton year.
        
        Return: int
        """
        return YEAR;
    
    @property
    def start_mon(self):
        """
        Return the EnergyPlus simulaton start month.
        
        Return: int
        """
        return self._eplus_run_st_mon;
    
    @property
    def start_day(self):
        """
        Return the EnergyPlus simulaton start day of the month.
        
        Return: int
        """
        return self._eplus_run_st_day;
    
    @property
    def start_weekday(self):
        """
        Return the EnergyPlus simulaton start weekday. 0 is Monday, 6 is Sunday.
        
        Return: int
        """
        return self._eplus_run_st_weekday;

    @property
    def env_name(self):
        """
        Return the environment name.

        Return: string
        """
        return self._env_name;
