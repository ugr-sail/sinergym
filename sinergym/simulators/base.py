"""
Base simulator class.
"""
from typing import Union

import numpy as np


class BaseSimulator(object):
    """Abstract class for defining a simulation engine."""

    def __init__(self):
        """Class constructor."""
        pass

    def start_simulation(self) -> bool:
        """Start a new simulation."""
        raise NotImplementedError

    def stop_simulation(self) -> bool:
        """Stop the current simulation."""
        raise NotImplementedError

    def send_action(self, action: Union[int, np.ndarray]) -> bool:
        """Send a control action to the simulator."""
        raise NotImplementedError

    def receive_observation(self) -> bool:
        """Method for reading an observation from the simulator."""
        raise NotImplementedError
