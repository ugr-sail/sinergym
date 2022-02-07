"""Renewed EnergyPlus connection interface."""

import os
import socket
from datetime import datetime
from typing import Any, Tuple

from eppy.modeleditor import IDF

from .base import BaseSimulator


class EnergyPlus(BaseSimulator):

    def __init__(
            self,
            idf_file: str,
            weather_file: str,
            variables_file: str,
            env_name: str):
        """EnergyPlus simulator connector.

        Args:
            idf_file (str): IDF file with the building model.
            weather_file (str): EPW file with weather data.
            variables_file (str): Configuration file with the variables used in the simulation.
            env_name (str): Name of the environment.

        Raises:
            KeyError: the environment variable BCVTB_PATH has not been defined.
            KeyError: the environment variable EPLUS_PATH has not been defined.
        """

        # Access BCVTB and EnergyPlus locations
        try:
            self.bcvtb_path = os.environ['BCVTB_PATH']
        except BaseException:
            raise KeyError('BCVTB_PATH environment variable not set.')
        try:
            self.eplus_path = os.environ['EPLUS_PATH']
        except BaseException:
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

    def start_simulation(self) -> bool:
        """Starts the simulation."""
        return True

    def end_simulation(self) -> bool:
        """Ends the simulation."""
        return True

    def send_action(self) -> bool:
        """Sends a new action to the simulator."""
        return True

    def receive_observation(self) -> bool:
        """Receive a new observation from the environment."""
        return True

    def _create_socket(self) -> Tuple[Any, str, str]:
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

    def _get_run_period(self) -> int:
        """Get the length of the run in timesteps."""

        self.timestep = self.idf.idfobjects['Timestep'][0]['Number_of_Timesteps_per_Hour']
        self.start_date = datetime(
            1991,
            self.idf.idfobjects['RunPeriod'][0]['Begin_Month'],
            self.idf.idfobjects['RunPeriod'][0]['Begin_Day_of_Month'])
        self.final_date = datetime(
            1991,
            self.idf.idfobjects['RunPeriod'][0]['End_Month'],
            self.idf.idfobjects['RunPeriod'][0]['End_Day_of_Month'])
        duration = (self.final_date -
                    self.start_date).total_seconds() / 3600  # hours
        times_to_repeat = self.idf.idfobjects['RunPeriod'][0]['Number_of_Times_Runperiod_to_be_Repeated']
        timesteps = duration * times_to_repeat * self.timestep
        return int(timesteps)
