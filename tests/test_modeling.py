import json
import os

import pytest
from epw.weather import Weather

from sinergym.utils.constants import WEEKDAY_ENCODING

# ---------------------------------------------------------------------------- #
#                    Variables and Building model adaptation                   #
# ---------------------------------------------------------------------------- #


def test_adapt_building_to_epw(model_5zone):
    # Read the current Location and DesignDays
    locations = model_5zone.building['Site:Location']
    designdays = model_5zone.building['SizingPeriod:DesignDay']
    assert len(locations) == 1
    assert len(designdays) == 2

    # Check old location is correct
    assert locations.get(
        'CHICAGO_IL_USA TMY2-94846',
        False)
    location = locations['CHICAGO_IL_USA TMY2-94846']
    assert isinstance(location.get('latitude'), float)
    assert isinstance(location.get('longitude'), float)
    assert isinstance(location.get('time_zone'), float)
    assert isinstance(location.get('elevation'), float)

    # Check old Designday is correct
    assert designdays.get(
        'CHICAGO_IL_USA Annual Heating 99% Design Conditions DB', False)
    winter_day = designdays['CHICAGO_IL_USA Annual Heating 99% Design Conditions DB']
    assert winter_day['day_type'] == 'WinterDesignDay'

    assert designdays.get(
        'CHICAGO_IL_USA Annual Cooling 1% Design Conditions DB/MCWB', False)
    summer_day = designdays['CHICAGO_IL_USA Annual Cooling 1% Design Conditions DB/MCWB']
    assert summer_day['day_type'] == 'SummerDesignDay'

    # Existing DesignDay names
    winterday = 'Ann Htg 99.6% Condns DB'
    summerday = 'Ann Clg .4% Condns DB=>MWB'
    # Adapt building to epw
    model_5zone.adapt_building_to_epw(summerday=summerday, winterday=winterday)

    # we do the same and check new values for Location and DesignDays
    locations = model_5zone.building['Site:Location']
    designdays = model_5zone.building['SizingPeriod:DesignDay']
    assert len(locations) == 1
    assert len(designdays) == 2
    # Check new location is correct
    assert locations.get('davis monthan afb_az_usa design_conditions', False)
    location = locations['davis monthan afb_az_usa design_conditions']
    assert isinstance(location.get('latitude'), float)
    assert isinstance(location.get('longitude'), float)
    assert isinstance(location.get('time_zone'), float)
    assert isinstance(location.get('elevation'), float)

    # Check new DesignDay is correct
    assert designdays.get('davis monthan afb ann htg 99.6% condns db', False)
    winter_day = designdays['davis monthan afb ann htg 99.6% condns db']
    assert winter_day['day_type'] == 'WinterDesignDay'
    assert winter_day.get('month', False)
    assert winter_day.get('day_of_month', False)
    assert winter_day.get('maximum_dry_bulb_temperature', False)
    assert winter_day.get('barometric_pressure', False)
    assert winter_day.get('wind_speed', False)

    assert designdays.get(
        'davis monthan afb ann clg .4% condns db=>mwb', False)
    summer_day = designdays['davis monthan afb ann clg .4% condns db=>mwb']
    assert summer_day['day_type'] == 'SummerDesignDay'
    assert winter_day.get('month', False)
    assert winter_day.get('day_of_month', False)
    assert winter_day.get('maximum_dry_bulb_temperature', False)
    assert winter_day.get('barometric_pressure', False)
    assert winter_day.get('wind_speed', False)


def test_adapt_building_to_variables(model_5zone, building_5zone):
    # State before method
    assert not building_5zone.get('Output:Variable', False)
    # State after method
    assert len(
        model_5zone.building['Output:Variable']) == len(
        model_5zone._variables)
    # Check all variable names are correct
    original_varible_names = [
        variable for variable, _ in list(
            model_5zone._variables.values())]
    for output_variable in list(
            model_5zone.building['Output:Variable'].values()):
        assert output_variable['variable_name'] in original_varible_names


def test_adapt_building_to_meters(model_5zone, building_5zone):
    # State before method
    assert not building_5zone.get('Output:Meter', False)
    # State after method
    assert len(
        model_5zone.building['Output:Meter']) == len(
        model_5zone._meters)
    # Check all meter names are correct
    original_meter_names = [
        meter for meter in list(
            model_5zone._meters.values())]
    for output_meter in list(
            model_5zone.building['Output:Meter'].values()):
        assert output_meter['key_name'] in original_meter_names


def test_adapt_building_to_config(model_5zone, building_5zone):
    # Check default config
    assert list(building_5zone['Timestep'].values())[
        0]['number_of_timesteps_per_hour'] == 4

    # Check new config
    assert list(model_5zone.building['Timestep'].values())[
        0]['number_of_timesteps_per_hour'] == 2

    # Check runperiod elements is the same than runperiod from config specified
    config_runperiod = model_5zone.building_config['runperiod']
    runperiod = model_5zone.runperiod
    for i, key in enumerate(
            ['start_day', 'start_month', 'start_year', 'end_day', 'end_month', 'end_year']):
        assert config_runperiod[i] == runperiod[key]


def test_save_building_model(model_5zone):
    assert model_5zone.episode_path is None
    # Create episode path before save (else a exception will happen)
    model_5zone.set_episode_working_dir()
    assert model_5zone.episode_path is not None
    # save current model
    path_save = model_5zone.save_building_model()
    # Read the path save and check building model is saved
    building = {}
    with open(path_save) as json_f:
        building = json.load(json_f)
    assert len(building) > 0
    # Check save runtime error if path is not specified
    model_5zone.episode_path = None
    with pytest.raises(RuntimeError):
        model_5zone.save_building_model()
    # Update episode path to wrong value to check save error
    model_5zone.episode_path = 'unknown_path/unexistent_dir'
    with pytest.raises(OSError):
        model_5zone.save_building_model()

# ---------------------------------------------------------------------------- #
#                        EPW and Weather Data management                       #
# ---------------------------------------------------------------------------- #


def test_check_model_wrong_weather(model_5zone_several_weathers):
    model_5zone_several_weathers._check_eplus_config()
    # update weather paths with one which does not exist
    model_5zone_several_weathers.weather_files.append('unknown_weather.epw')
    with pytest.raises(FileNotFoundError):
        model_5zone_several_weathers._check_eplus_config()


def test_check_model_wrong_config(model_5zone_several_weathers):
    model_5zone_several_weathers._check_eplus_config()
    # update config dictionary with wrong values
    if model_5zone_several_weathers.building_config.get('timesteps_per_hour'):
        model_5zone_several_weathers.building_config['timesteps_per_hour'] = -7
        with pytest.raises(ValueError):
            model_5zone_several_weathers._check_eplus_config()
        model_5zone_several_weathers.building_config['timesteps_per_hour'] = 2
    if model_5zone_several_weathers.building_config.get('runperiod'):
        model_5zone_several_weathers.building_config['runperiod'] = (
            1, 2, 3, 4)
        with pytest.raises(ValueError):
            model_5zone_several_weathers._check_eplus_config()
        model_5zone_several_weathers.building_config['runperiod'] = 2024
        with pytest.raises(TypeError):
            model_5zone_several_weathers._check_eplus_config()
        model_5zone_several_weathers.building_config['runperiod'] = (
            1, 2, 1993, 2, 3, 1993)
    model_5zone_several_weathers.building_config['Unknown_option'] = 100
    # It will be ignored by sinergym
    model_5zone_several_weathers._check_eplus_config()


def test_update_weather_path(model_5zone_several_weathers):
    # Check that we have more than one weather file
    assert len(model_5zone_several_weathers.weather_files) > 1
    # Check there is one specified before and after update weather path
    assert model_5zone_several_weathers._weather_path is not None
    model_5zone_several_weathers.update_weather_path()
    assert model_5zone_several_weathers._weather_path is not None
    # Try update without any weather
    model_5zone_several_weathers.weather_files = []
    with pytest.raises(ValueError):
        model_5zone_several_weathers.update_weather_path()
    # Try using a wrong weather file
    model_5zone_several_weathers.weather_files = ['unknown_weather.epw']
    with pytest.raises(FileNotFoundError):
        model_5zone_several_weathers.update_weather_path()


def test_apply_weather_variability(model_5zone):
    # First set a episode dir in execution
    assert model_5zone.episode_path is None
    model_5zone.set_episode_working_dir()
    assert model_5zone.episode_path is not None
    # Check apply None variation return original weather_path
    path_result = model_5zone.apply_weather_variability(
        weather_variability=None)
    original_filename = model_5zone._weather_path.split('/')[-1]
    path_filename = path_result.split('/')[-1]
    assert original_filename == path_filename
    # Check with a variation
    weather_variability = {
        'Dry Bulb Temperature': (1.0, 0.0, 24.0),
        'Wind Speed': (3.0, 0.0, 35.0)
    }
    path_result = model_5zone.apply_weather_variability(
        weather_variability=weather_variability)
    filename = model_5zone._weather_path.split('/')[-1]
    filename = filename.split('.epw')[0]
    filename += '_OU_Noise.epw'
    path_expected = model_5zone.episode_path + '/' + filename
    assert path_result == path_expected
    # Check that path exists
    assert os.path.exists(path_result)
    # Lets load the original weather file and the new one
    original = Weather()
    original.read(model_5zone._weather_path)
    noise = Weather()
    noise.read(path_result)

    # Ensure dataframes are loaded
    assert original.dataframe is not None
    assert noise.dataframe is not None

    # Check that the noise is applied in drybulb and windspd columns and not
    # in others
    assert (original.dataframe['Dry Bulb Temperature'] !=
            noise.dataframe['Dry Bulb Temperature']).any()
    assert (original.dataframe['Wind Speed'] !=
            noise.dataframe['Wind Speed']).any()
    assert (original.dataframe['Relative Humidity'] ==
            noise.dataframe['Relative Humidity']).all()
    assert (original.dataframe['Wind Direction'] ==
            noise.dataframe['Wind Direction']).all()

# ---------------------------------------------------------------------------- #
#                          Schedulers info extraction                          #
# ---------------------------------------------------------------------------- #


def test_get_schedulers(model_5zone):
    # Testing scheduler attribute structure is correct
    assert len(model_5zone.schedulers) == len(
        model_5zone.building['Schedule:Compact'])
    for scheduler_name, definition in model_5zone.schedulers.items():
        assert isinstance(scheduler_name, str)
        assert isinstance(definition, dict)
        assert isinstance(definition['Type'], str)
        for key, value in definition.items():
            if key != 'Type':
                assert isinstance(value, dict)
                assert set(['field_name', 'table_name']
                           ) == set(value.keys())
                assert isinstance(
                    value['field_name'],
                    str) and isinstance(
                    value['table_name'],
                    str)
                assert key in list(
                    model_5zone.building[value['table_name']].keys())

# ---------------------------------------------------------------------------- #
#                           Runperiod info extraction                          #
# ---------------------------------------------------------------------------- #


def test_get_eplus_runperiod(model_5zone):
    runperiod = model_5zone.get_eplus_runperiod()
    building_runperiod = list(model_5zone.building['RunPeriod'].values())[0]

    assert runperiod['start_day'] == building_runperiod['begin_day_of_month']
    assert runperiod['start_month'] == building_runperiod['begin_month']
    assert runperiod['start_year'] == building_runperiod['begin_year']
    assert runperiod['end_day'] == building_runperiod['end_day_of_month']
    assert runperiod['end_month'] == building_runperiod['end_month']
    assert runperiod['end_year'] == building_runperiod['end_year']
    assert runperiod['start_weekday'] == WEEKDAY_ENCODING[building_runperiod['day_of_week_for_start_day'].lower()]
    assert runperiod['n_steps_per_hour'] == list(
        model_5zone.building['Timestep'].values())[0]['number_of_timesteps_per_hour']

# ---------------------------------------------------------------------------- #
#                  Working Folder for Simulation Management                    #
# ---------------------------------------------------------------------------- #


def test_set_episode_working_dir(model_5zone):
    # Check config has no episode path set up yet
    assert model_5zone.episode_path is None
    # Creating episode dir
    episode_path = model_5zone.set_episode_working_dir()
    # Check if new episode dir exists
    assert os.path.isdir(episode_path)
    # Check if workspace_path is none raise an error
    model_5zone.workspace_path = None
    with pytest.raises(Exception):
        model_5zone.set_episode_working_dir()


def test_set_workspace_dir(model_5zone):
    # Check current config workspace dir and if exists
    current_workspace_path = model_5zone.workspace_path
    assert 'PYTESTCONFIG-res' in current_workspace_path
    assert os.path.isdir(current_workspace_path)
    # Set a new workspace_path
    new_workspace_path = model_5zone._set_workspace_dir(
        env_name='PYTESTCONFIG')
    # The name should be the same except last number id
    assert current_workspace_path[:-1] == new_workspace_path[:-1]
    assert int(current_workspace_path[-1]) < int(new_workspace_path[-1])
    # Check if new workspace path exists
    assert os.path.isdir(new_workspace_path)


def test_get_working_folder(model_5zone):
    expected = 'PYTESTCONFIG-res1/episode-1'
    parent_dir = 'PYTESTCONFIG-res1'
    dir_sig = 'episode-'
    path = model_5zone._get_working_folder(parent_dir, dir_sig)
    assert expected == path


def test_rm_past_history_dir(model_5zone):
    # Check num of dir in workspace path is less than 10
    n_dir = len([i for i in os.listdir(model_5zone.workspace_path)
                if os.path.isdir(os.path.join(model_5zone.workspace_path, i))])
    assert n_dir < 10
    # Create more than 10 episodes dir
    for _ in range(15):
        model_5zone.set_episode_working_dir()
    # Check number of dirs is 10 (no more)
    n_dir = len([i for i in os.listdir(model_5zone.workspace_path)
                if os.path.isdir(os.path.join(model_5zone.workspace_path, i))])
    assert n_dir == 10
