import pytest
import sinergym.utils.config as config
from opyplus import Epm


def test_get_eplus_run_info(config):
    info = config._get_eplus_run_info()
    assert info == (1, 1, 0, 3, 31, 0, 0, 4)


def test_get_one_epi_len(config):
    total_time = config._get_one_epi_len()
    assert total_time == 7776000


def test_get_working_folder(config):
    expected = 'Eplus-env-TESTCONFIG-res1/Eplus-env-sub_run1'
    parent_dir = 'Eplus-env-TESTCONFIG-res1'
    dir_sig = '-sub_run'
    path = config._get_working_folder(parent_dir, dir_sig)
    assert expected == path


def test_adapt_idf_to_epw(config):
    # Existing designday names
    winterday = 'Ann Htg 99.6% Condns DB'
    summerday = 'Ann Clg .4% Condns DB=>MWB'
    # Read the current Location and DesignDays
    location = config.building.site_location[0]
    winter_day = config.building.sizingperiod_designday.select(
        lambda designday: winterday.lower() in designday.name.lower())[0]
    summer_day = config.building.sizingperiod_designday.select(
        lambda designday: summerday.lower() in designday.name.lower())[0]

    # Check old location is correct
    assert location.name.lower(
    ) == 'Pittsburgh Allegheny Co Ap_PA_USA Design_Conditions'.lower()
    assert float(location.latitude) == 40.35
    assert float(location.longitude) == -79.92
    assert float(location.time_zone) == -5.00
    assert float(location.elevation) == 380.00

    # Check old Designday is correct
    assert winter_day.name.lower(
    ) == 'Pittsburgh Allegheny Co Ap Ann Htg 99.6% Condns DB'.lower()
    assert winter_day.day_type.lower() == 'WinterDesignDay'.lower()
    assert int(winter_day.month) == 1
    assert int(winter_day.day_of_month) == 21
    assert float(winter_day.maximum_dry_bulb_temperature) == -15.4
    assert float(winter_day.barometric_pressure) == 96842.0
    assert float(winter_day.wind_speed) == 4.7

    assert summer_day.name.lower(
    ) == 'Pittsburgh Allegheny Co Ap Ann Clg .4% Condns DB=>MWB'.lower()
    assert summer_day.day_type.lower() == 'SummerDesignDay'.lower()
    assert int(summer_day.month) == 7
    assert int(summer_day.day_of_month) == 21
    assert float(summer_day.maximum_dry_bulb_temperature) == 32.2
    assert float(summer_day.barometric_pressure) == 96842.0
    assert float(summer_day.wind_speed) == 4.4

    # Adapt idf to epw
    config.adapt_idf_to_epw(summerday=summerday, winterday=winterday)

    # we do the same and check new values for Location and Designdays
    location = config.building.site_location[0]
    winter_day = config.building.sizingperiod_designday.select(
        lambda designday: winterday.lower() in designday.name.lower())[0]
    summer_day = config.building.sizingperiod_designday.select(
        lambda designday: summerday.lower() in designday.name.lower())[0]
    # Check new location is correct
    assert location.name.lower(
    ) == 'tucscon-davis-monthan afb_az_usa design_conditions'.lower()
    assert float(location.latitude) == 32.17
    assert float(location.longitude) == -110.88
    assert float(location.time_zone) == -7.0
    assert float(location.elevation) == 809.0

    # Check new Designday is correct
    assert winter_day.name.lower(
    ) == 'tucscon-davis-monthan afb ann htg 99.6% condns db'.lower()
    assert winter_day.day_type.lower() == 'winterdesignday'.lower()
    assert int(winter_day.month) == 12
    assert int(winter_day.day_of_month) == 21
    assert float(winter_day.maximum_dry_bulb_temperature) == 0.2
    assert float(winter_day.barometric_pressure) == 91976.0
    assert float(winter_day.wind_speed) == 2.4

    assert summer_day.name.lower(
    ) == 'tucscon-davis-monthan afb ann clg .4% condns db=>mwb'.lower()
    assert summer_day.day_type.lower() == 'summerdesignday'.lower()
    assert int(summer_day.month) == 7
    assert int(summer_day.day_of_month) == 21
    assert float(summer_day.maximum_dry_bulb_temperature) == 40.7
    assert float(summer_day.barometric_pressure) == 91976.0
    assert float(summer_day.wind_speed) == 5.0


def test_apply_extra_conf(config):
    # Check default config
    assert int(config.building.timestep[0].number_of_timesteps_per_hour) == 4

    # Set new extra configuration
    config.apply_extra_conf()

    # Check new config
    assert int(config.building.timestep[0].number_of_timesteps_per_hour) == 2
