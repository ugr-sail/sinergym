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

    def start_simulation(self) -> None:
        """Start a new simulation."""
        raise NotImplementedError

    def stop_simulation(self) -> None:
        """Stop the current simulation."""
        raise NotImplementedError

    def send_action(self, action: Union[int, np.ndarray]) -> None:
        """Send a control action to the simulator."""
        raise NotImplementedError

    def receive_observation(self) -> None:
        """Method for reading an observation from the simulator."""
        raise NotImplementedError
