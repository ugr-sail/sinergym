"""
Class for connecting EnergyPlus with Python using Ptolomy server
Author: Javier Jiménez based on Zhiang Zhang's implementation
https://github.com/zhangzhizza/Gym-Eplus
"""

import os
import socket
from eppy.modeleditor import IDF
from datetime import datetime


class EnergyPlus(object):
    """"""

    def __init__(self, idf_file, weather_file, variables_file, env_name):
        """"""

        # Access BCVTB and EnergyPlus locations
        try:
            self.bcvtb_path = os.environ['BCVTB_PATH']
        except:
            raise KeyError('BCVTB_PATH environment variable not set.')
        try:
            self.eplus_path = os.environ['EPLUS_PATH']
        except:
            raise KeyError('EPLUS_PATH environment variable not set.')

        # Read IDF and weather files
        IDF.setiddname(os.path.join(self.eplus_path, 'Energy+.idd'))
        self.idf = IDF(idf_file)
        self.idf_file = idf_file
        self.weather_file = weather_file
        # Max number of timesteps
        self.max_timesteps = self._get_run_period()
        self.run_number = 0
        # Create output folder
        self.env_name = env_name
        self.output_folder = env_name + '_' + datetime.now().strftime('%Y%m%d%H%M')
        os.makedirs(self.output_folder, exist_ok=True)
        # Create socket for communication with EnergyPlus
        self._socket, self._host, self._port = self._create_socket()

    def start_simulation(self):
        """"""
        return True

    def end_simulation(self):
        """"""
        return True

    def send_action(self):
        """"""
        return True

    def return_observation(self):
        """"""
        return True

    def _create_socket(self):
        """Create socket, host and port."""

        s = socket.socket()
        # Get local machine name
        host = socket.gethostname()
        # Bind to the host and any available port
        s.bind((host, 0))
        sockname = s.getsockname()
        # Get the port number
        port = sockname[1]
        # Listen on request
        s.listen(60)
        return s, host, port

    def _get_run_period(self):
        """Get the length of the run in timesteps."""

        self.timestep = self.idf.idfobjects['Timestep'][0]['Number_of_Timesteps_per_Hour']
        self.start_date = datetime(1991, 
            self.idf.idfobjects['RunPeriod'][0]['Begin_Month'], 
            self.idf.idfobjects['RunPeriod'][0]['Begin_Day_of_Month']
        )
        self.final_date = datetime(1991, 
            self.idf.idfobjects['RunPeriod'][0]['End_Month'], 
            self.idf.idfobjects['RunPeriod'][0]['End_Day_of_Month']
        )
        duration = (self.final_date - self.start_date).total_seconds() / 3600 # hours
        times_to_repeat = self.idf.idfobjects['RunPeriod'][0]['Number_of_Times_Runperiod_to_be_Repeated']
        timesteps = duration * times_to_repeat * self.timestep
        return int(timesteps)