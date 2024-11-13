import warnings
import pytest

# Ignore epw module warning (epw module mistake)
warnings.filterwarnings(
    "ignore",
    module='epw')
