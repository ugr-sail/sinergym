import pytest
import sinergym.utils.config as config


def test_get_eplus_run_info(simulator):
    info = simulator._config._get_eplus_run_info()
    assert info == (1, 1, 0, 3, 31, 0, 0, 4)
