"""
Base simulator class.
"""


import os
import numpy as np


class BaseSimulator(object):
    """Abstract class for defining a simulation engine."""

    def __init__(self):
        """Class constructor."""
        pass

    def start_simulation(self):
        """Start a new simulation."""
        raise NotImplementedError

    def stop_simulation(self):
        """Stop the current simulation."""
        raise NotImplementedError

    def send_action(self, action):
        """Send a control action to the simulator."""
        raise NotImplementedError

    def receive_observation(self):
        """Method for reading an observation from the simulator."""
        raise NotImplementedError
