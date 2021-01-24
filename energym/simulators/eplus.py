"""
Class for connecting EnergyPlus with Python using Ptolomy server
Author: Javier Jiménez based on Zhiang Zhang's implementation
https://github.com/zhangzhizza/Gym-Eplus
"""


import os
import threading
import socket


class EnergyPlus(object):
    """"""

    def __init__(self):
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

        # Create socket
        self._socket, self._host, self._port = self._create_socket()

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