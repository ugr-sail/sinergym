import os

import pytest
from opyplus import Epm, Idd

import sinergym.utils.config as config


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
    ) == 'davis monthan afb_az_usa design_conditions'.lower()
    assert float(location.latitude) == 32.17
    assert float(location.longitude) == -110.88
    assert float(location.time_zone) == -7.0
    assert float(location.elevation) == 809.0

    # Check new Designday is correct
    assert winter_day.name.lower(
    ) == 'davis monthan afb ann htg 99.6% condns db'.lower()
    assert winter_day.day_type.lower() == 'winterdesignday'.lower()
    assert int(winter_day.month) == 12
    assert int(winter_day.day_of_month) == 21
    assert float(winter_day.maximum_dry_bulb_temperature) == 0.5
    assert float(winter_day.barometric_pressure) == 91976.0
    assert float(winter_day.wind_speed) == 2.1

    assert summer_day.name.lower(
    ) == 'davis monthan afb ann clg .4% condns db=>mwb'.lower()
    assert summer_day.day_type.lower() == 'summerdesignday'.lower()
    assert int(summer_day.month) == 7
    assert int(summer_day.day_of_month) == 21
    assert float(summer_day.maximum_dry_bulb_temperature) == 40.8
    assert float(summer_day.barometric_pressure) == 91976.0
    assert float(summer_day.wind_speed) == 4.8


def test_apply_extra_conf(config):
    # Check default config
    assert int(config.building.timestep[0].number_of_timesteps_per_hour) == 4

    # Set new extra configuration
    config.apply_extra_conf()

    # Check new config
    assert int(config.building.timestep[0].number_of_timesteps_per_hour) == 2


def test_save_building_model(config, eplus_path, idf_path):
    assert config.episode_path is None
    # Create episode path before save (else a exception will happen)
    config.set_episode_working_dir()
    assert config.episode_path is not None
    # save current model
    path_save = config.save_building_model()
    # Read the path save idf and check IDF saved
    idd = Idd(os.path.join(eplus_path, 'Energy+.idd'))
    building = Epm.from_idf(idf_path, idd_or_version=idd)
    assert (building.get_info() is not None) or (building.get_info() != '')


def test_apply_weather_variability(config):
    # First set a epìsode dir in experiment
    assert config.episode_path is None
    config.set_episode_working_dir()
    assert config.episode_path is not None
    # Check apply None variation return original weather_path
    path_result = config.apply_weather_variability(variation=None)
    assert path_result == config._weather_path
    # Check with a variation
    variation = (1.0, 0.0, 0.001)
    path_result = config.apply_weather_variability(variation=variation)
    filename = config._weather_path.split('/')[-1]
    filename = filename.split('.epw')[0]
    filename += '_Random_%s_%s_%s.epw' % (
        str(variation[0]), str(variation[1]), str(variation[2]))
    path_expected = config.episode_path + '/' + filename
    assert path_result == path_expected
    assert os.path.exists(path_result)


def test_get_eplus_run_info(config):
    info = config._get_eplus_run_info()
    assert info == (1, 1, 0, 12, 31, 0, 0, 4)


def test_get_one_epi_len(config):
    total_time = config._get_one_epi_len()
    assert total_time == 31536000


def test_set_experiment_working_dir(config):
    # Check current config experiment working dir and if exists
    current_experiment_path = config.experiment_path
    assert 'Eplus-env-TESTCONFIG-res' in current_experiment_path
    assert os.path.isdir(current_experiment_path)
    # Set a new experiment_path
    new_experiment_path = config.set_experiment_working_dir(
        env_name='TESTCONFIG')
    # The name should be the same except last number id
    assert current_experiment_path[:-1] == new_experiment_path[:-1]
    assert int(current_experiment_path[-1]) < int(new_experiment_path[-1])
    # Check if new experiment path exists
    assert os.path.isdir(new_experiment_path)


def test_set_episode_working_dir(config):
    # Check config has no episode path set up yet
    assert config.episode_path is None
    # Creating episode dir
    episode_path = config.set_episode_working_dir()
    # Check if new episode dir exists
    assert os.path.isdir(episode_path)


def test_get_working_folder(config):
    expected = 'Eplus-env-TESTCONFIG-res1/Eplus-env-sub_run1'
    parent_dir = 'Eplus-env-TESTCONFIG-res1'
    dir_sig = '-sub_run'
    path = config._get_working_folder(parent_dir, dir_sig)
    assert expected == path


def test_rm_past_history_dir(config):
    # Check num of dir in experiment path is less than 10
    n_dir = len([i for i in os.listdir(config.experiment_path)
                 if os.path.isdir(os.path.join(config.experiment_path, i))])
    assert n_dir < 10
    # Create more than 10 episodes dir
    for _ in range(15):
        config.set_episode_working_dir()
    # Check number of dirs is 10 (no more)
    n_dir = len([i for i in os.listdir(config.experiment_path)
                 if os.path.isdir(os.path.join(config.experiment_path, i))])
    assert n_dir == 10
