import pytest
import sinergym.utils.config as config


def test_get_eplus_run_info(simulator):
    info = simulator._config._get_eplus_run_info()
    assert info == (1, 1, 0, 3, 31, 0, 0, 4)


def test_get_one_epi_len(simulator):
    total_time = simulator._config._get_one_epi_len()
    assert total_time == simulator._eplus_one_epi_len
    assert total_time == 7776000
