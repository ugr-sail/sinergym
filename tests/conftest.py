import json
import os
import shutil
from glob import glob  # to find directories with patterns
from importlib import resources

import pytest
import yaml
from epw.weather import Weather

import sinergym
from sinergym.config.modeling import ModelJSON
from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.constants import *
from sinergym.utils.controllers import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

# ---------------------------------------------------------------------------- #
#                                Root Directory                                #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def sinergym_path():
    sinergym_resource = resources.files('sinergym')
    return os.path.abspath(os.path.join(str(sinergym_resource), os.pardir))

# ---------------------------------------------------------------------------- #
#                                     Paths                                    #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def pkg_data_path():
    return PKG_DATA_PATH


@pytest.fixture(scope='function')
def pkg_mock_path():
    mock_resource = resources.files('tests') / 'mock'
    return str(mock_resource)


@pytest.fixture(scope='function')
def json_path_5zone(pkg_data_path):
    return os.path.join(pkg_data_path, 'buildings', '5ZoneAutoDXVAV.epJSON')


@pytest.fixture(scope='function')
def weather_path_pittsburgh(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw')

# ---------------------------------------------------------------------------- #
#                         Default Environment Arguments                        #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def TIME_VARIABLES():
    return ['month', 'day_of_month', 'hour']


@pytest.fixture(scope='function')
def ACTION_SPACE_5ZONE():
    return gym.spaces.Box(
        low=np.array([12.0, 23.25], dtype=np.float32),
        high=np.array([23.25, 30.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32
    )


@pytest.fixture(scope='function')
def VARIABLES_5ZONE():
    return {
        'outdoor_temperature': (
            'Site Outdoor Air DryBulb Temperature',
            'Environment'),
        'outdoor_humidity': (
            'Site Outdoor Air Relative Humidity',
            'Environment'),
        'wind_speed': (
            'Site Wind Speed',
            'Environment'),
        'wind_direction': (
            'Site Wind Direction',
            'Environment'),
        'diffuse_solar_radiation': (
            'Site Diffuse Solar Radiation Rate per Area',
            'Environment'),
        'direct_solar_radiation': (
            'Site Direct Solar Radiation Rate per Area',
            'Environment'),
        'htg_setpoint': (
            'Zone Thermostat Heating Setpoint Temperature',
            'SPACE5-1'),
        'clg_setpoint': (
            'Zone Thermostat Cooling Setpoint Temperature',
            'SPACE5-1'),
        'air_temperature': (
            'Zone Air Temperature',
            'SPACE5-1'),
        'air_humidity': (
            'Zone Air Relative Humidity',
            'SPACE5-1'),
        'people_occupant': (
            'Zone People Occupant Count',
            'SPACE5-1'),
        'co2_emission': (
            'Environmental Impact Total CO2 Emissions Carbon Equivalent Mass',
            'site'),
        'HVAC_electricity_demand_rate': (
            'Facility Total HVAC Electricity Demand Rate',
            'Whole Building')
    }


@pytest.fixture(scope='function')
def METERS_5ZONE():
    return {'total_electricity_HVAC': 'Electricity:HVAC'}


@pytest.fixture(scope='function')
def ACTUATORS_5ZONE():
    return {
        'Heating_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'HTG-SETP-SCH'),
        'Cooling_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'CLG-SETP-SCH')
    }


@pytest.fixture(scope='function')
def CONTEXT_5ZONE():
    return {
        'Occupancy': (
            'Schedule:Compact',
            'Schedule Value',
            'OCCUPY-1')
    }


@pytest.fixture(scope='function')
def INITIAL_CONTEXT():
    return [1.0]


@pytest.fixture(scope='function')
def ACTION_SPACE_DISCRETE_5ZONE():
    return gym.spaces.Discrete(10)


@pytest.fixture(scope='function')
def ACTION_SPACE_DATACENTER():
    return gym.spaces.Box(
        low=np.array([15.0, 22.0], dtype=np.float32),
        high=np.array([22.0, 30.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32
    )


@pytest.fixture(scope='function')
def VARIABLES_DATACENTER():
    return {
        'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
        'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
        'wind_speed': ('Site Wind Speed', 'Environment'),
        'wind_direction': ('Site Wind Direction', 'Environment'),
        'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
        'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
        'west_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'West Zone'),
        'east_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'East Zone'),
        'west_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'West Zone'),
        'east_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'East Zone'),
        'west_zone_air_temperature': ('Zone Air Temperature', 'West Zone'),
        'east_zone_air_temperature': ('Zone Air Temperature', 'East Zone'),
        'west_zone_thermal_comfort_mean_radiant_temperature': ('Zone Thermal Comfort Mean Radiant Temperature', 'West Zone PEOPLE'),
        'east_zone_thermal_comfort_mean_radiant_temperature': ('Zone Thermal Comfort Mean Radiant Temperature', 'East Zone PEOPLE'),
        'west_zone_air_humidity': ('Zone Air Relative Humidity', 'West Zone'),
        'east_zone_air_humidity': ('Zone Air Relative Humidity', 'East Zone'),
        'west_zone_thermal_comfort_clothing_value': ('Zone Thermal Comfort Clothing Value', 'West Zone PEOPLE'),
        'east_zone_thermal_comfort_clothing_value': ('Zone Thermal Comfort Clothing Value', 'East Zone PEOPLE'),
        'west_zone_thermal_comfort_fanger_model_ppd': ('Zone Thermal Comfort Fanger Model PPD', 'West Zone PEOPLE'),
        'east_zone_thermal_comfort_fanger_model_ppd': ('Zone Thermal Comfort Fanger Model PPD', 'East Zone PEOPLE'),
        'west_zone_people_occupant': ('Zone People Occupant Count', 'West Zone'),
        'east_zone_people_occupant': ('Zone People Occupant Count', 'East Zone'),
        'west_zone_people_air_temperature': ('People Air Temperature', 'West Zone PEOPLE'),
        'east_zone_people_air_temperature': ('People Air Temperature', 'East Zone PEOPLE'),
        'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
    }


@pytest.fixture(scope='function')
def METERS_DATACENTER():
    return {}


@pytest.fixture(scope='function')
def ACTUATORS_DATACENTER():
    return {
        'Heating_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'Heating Setpoints'),
        'Cooling_Setpoint_RL': (
            'Schedule:Compact',
            'Schedule Value',
            'Cooling Setpoints')
    }

# ---------------------------------------------------------------------------- #
#                       Default environment configurations                     #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def conf_5zone(pkg_mock_path):
    conf_path = os.path.join(
        pkg_mock_path,
        'environment_configurations',
        '5ZoneAutoDXVAV.yaml')
    with open(conf_path, 'r') as yaml_conf:
        conf = yaml.safe_load(yaml_conf)
    return conf


@pytest.fixture(scope='function')
def conf_5zone_exceptions(pkg_mock_path):
    conf_exceptions = []
    for i in range(1, 6):
        conf_path = os.path.join(pkg_mock_path,
                                 'environment_configurations',
                                 '5ZoneAutoDXVAV_exception{}.yaml'.format(i))
        with open(conf_path, 'r') as yaml_conf:
            conf_exceptions.append(yaml.safe_load(yaml_conf))

    return conf_exceptions


# ---------------------------------------------------------------------------- #
#                                 Environments                                 #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def env_demo(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE,
        CONTEXT_5ZONE,
        INITIAL_CONTEXT):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        context=CONTEXT_5ZONE,
        initial_context=INITIAL_CONTEXT,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (1, 1, 1991, 31, 1, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_demo_energy_cost(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (1, 1, 1991, 31, 1, 1991)
        }
    )
    env = EnergyCostWrapper(
        env,
        energy_cost_data_path='/workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv')
    return env


@pytest.fixture(scope='function')
def env_demo_summer(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (7, 1, 1991, 31, 7, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_demo_summer_energy_cost(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (7, 1, 1991, 31, 7, 1991)
        }
    )
    env = EnergyCostWrapper(
        env,
        energy_cost_data_path='/workspaces/sinergym/sinergym/data/energy_cost/PVPC_active_energy_billing_Iberian_Peninsula_2023.csv')
    return env


@pytest.fixture(scope='function')
def env_demo_discrete(env_demo):
    return DiscretizeEnv(env=env_demo, discrete_space=gym.spaces.Discrete(
        10), action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION)


@pytest.fixture(scope='function')
def env_5zone(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE,
        CONTEXT_5ZONE,
        INITIAL_CONTEXT):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        context=CONTEXT_5ZONE,
        initial_context=INITIAL_CONTEXT,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env


@pytest.fixture(scope='function')
def env_5zone_stochastic(
        ACTION_SPACE_5ZONE,
        TIME_VARIABLES,
        VARIABLES_5ZONE,
        METERS_5ZONE,
        ACTUATORS_5ZONE):
    env = EplusEnv(
        building_file='5ZoneAutoDXVAV.epJSON',
        weather_files=[
            'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw'],
        action_space=ACTION_SPACE_5ZONE,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        actuators=ACTUATORS_5ZONE,
        weather_variability={
            'Dry Bulb Temperature': (
                1.0,
                0.0,
                24.0)},
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (
                1,
                1,
                1991,
                31,
                3,
                1991)})
    return env


@pytest.fixture(scope='function')
def env_datacenter(
        ACTION_SPACE_DATACENTER,
        TIME_VARIABLES,
        VARIABLES_DATACENTER,
        METERS_DATACENTER,
        ACTUATORS_DATACENTER):
    env = EplusEnv(
        building_file='2ZoneDataCenterHVAC_wEconomizer.epJSON',
        weather_files='USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        action_space=ACTION_SPACE_DATACENTER,
        time_variables=TIME_VARIABLES,
        variables=VARIABLES_DATACENTER,
        meters=METERS_DATACENTER,
        actuators=ACTUATORS_DATACENTER,
        reward=LinearReward,
        reward_kwargs={
            'temperature_variables': [
                'west_zone_air_temperature',
                'east_zone_air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        env_name='PYTESTGYM',
        building_config={
            'runperiod': (1, 1, 1991, 31, 3, 1991)
        }
    )
    return env

# ---------------------------------------------------------------------------- #
#                                  Simulators                                  #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def simulator_5zone(env_5zone):
    return env_5zone.energyplus_simulator


@pytest.fixture(scope='function')
def simulator_datacenter(env_datacenter):
    return env_datacenter.energyplus_simulator

# ---------------------------------------------------------------------------- #
#                               Modeling classes                               #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def building_5zone(json_path_5zone):
    with open(json_path_5zone) as json_f:
        building = json.load(json_f)
        return building


@pytest.fixture(scope='function')
def model_5zone(VARIABLES_5ZONE, METERS_5ZONE):

    return ModelJSON(
        env_name='PYTESTCONFIG',
        json_file='5ZoneAutoDXVAV.epJSON',
        weather_files=['USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        max_ep_store=10,
        building_config={
            'timesteps_per_hour': 2,
            'runperiod': (1, 2, 1993, 2, 3, 1993),
        })


@pytest.fixture(scope='function')
def model_5zone_several_weathers(
        VARIABLES_5ZONE,
        METERS_5ZONE):
    return ModelJSON(
        env_name='PYTESTCONFIG',
        json_file='5ZoneAutoDXVAV.epJSON',
        weather_files=[
            'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
            'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
        variables=VARIABLES_5ZONE,
        meters=METERS_5ZONE,
        max_ep_store=10,
        building_config={
            'timesteps_per_hour': 2,
            'runperiod': (
                1,
                2,
                1993,
                2,
                3,
                1993),
        })

# ---------------------------------------------------------------------------- #
#                          Environments with Wrappers                          #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def custom_logger_wrapper():
    class CustomLoggerWrapper(BaseLoggerWrapper):

        def __init__(
                self,
                env: gym.Env,
                logger_class: Callable = LoggerStorage):

            super(CustomLoggerWrapper, self).__init__(env, logger_class)
            # DEFINE CUSTOM VARIABLES AND SUMMARY VARIABLES
            self.custom_variables = ['custom_variable1', 'custom_variable2']
            self.summary_variables = [
                'episode_num',
                'double_mean_reward',
                'half_power_demand']

        # DEFINE ABSTRACT METHODS FOR METRICS CALCULATION

        def calculate_custom_metrics(self,
                                     obs: np.ndarray,
                                     action: Union[int, np.ndarray],
                                     reward: float,
                                     info: Dict[str, Any],
                                     terminated: bool,
                                     truncated: bool):
            # Variables combining information
            return [obs[0] * 2, obs[-1] + reward]

        def get_episode_summary(self) -> Dict[str, float]:
            # Get information from logger
            power_demands = [info['total_power_demand']
                             for info in self.data_logger.infos[1:]]

            # Data summary
            data_summary = {
                'episode_num': self.get_wrapper_attr('episode'),
                'double_mean_reward': np.mean(self.data_logger.rewards) * 2,
                'half_power_demand': np.mean(power_demands) / 2,
            }
            return data_summary

    return CustomLoggerWrapper


@pytest.fixture(scope='function')
def env_all_wrappers(env_demo):
    env = MultiObjectiveReward(
        env=env_demo,
        reward_terms=[
            'energy_term',
            'comfort_term'])
    env = PreviousObservationWrapper(env, previous_variables=[
        'htg_setpoint',
        'clg_setpoint',
        'air_temperature'])
    env = DatetimeWrapper(env)
    env = DiscreteIncrementalWrapper(
        env, initial_values=[
            21.0, 25.0], delta_temp=2, step_temp=0.5)
    env = NormalizeObservation(
        env=env)
    env = LoggerWrapper(env=env)
    env = CSVLogger(env=env)
    env = ReduceObservationWrapper(
        env=env,
        obs_reduction=[
            'outdoor_temperature',
            'outdoor_humidity',
            'air_temperature'])
    env = MultiObsWrapper(env=env, n=5, flatten=True)
    return env

# ---------------------------------------------------------------------------- #
#                                  Controllers                                 #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def random_controller(env_5zone):
    return RandomController(env=env_5zone)


@pytest.fixture(scope='function')
def zone5_controller(env_5zone):
    return RBC5Zone(env=env_5zone)


@pytest.fixture(scope='function')
def datacenter_controller(env_datacenter):
    return RBCDatacenter(env=env_datacenter)


@pytest.fixture(scope='function')
def datacenter_incremental_controller(env_datacenter):
    return RBCIncrementalDatacenter(env=env_datacenter)

# ---------------------------------------------------------------------------- #
#                      Building and weather python models                      #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def building(json_path_5zone):
    with open(json_path_5zone) as json_f:
        building_model = json.load(json_f)
    return building_model


@pytest.fixture(scope='function')
def weather_data(weather_path_pittsburgh):
    weather_data = Weather()
    weather_data.read(weather_path_pittsburgh)
    return weather_data

# ---------------------------------------------------------------------------- #
#                                    Rewards                                   #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def base_reward():
    return BaseReward()


@pytest.fixture(scope='function')
def custom_reward():
    class CustomReward(BaseReward):
        def __init__(self):
            super(CustomReward, self).__init__()

        def __call__(self):
            return -1.0, {}

    return CustomReward()


@pytest.fixture(scope='function')
def linear_reward():
    return LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@pytest.fixture(scope='function')
def energy_cost_linear_reward():
    return EnergyCostLinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        energy_cost_variables=['energy_cost'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0),
        temperature_weight=0.4,
        energy_weight=0.4,
        lambda_energy=1e-4,
        lambda_temperature=1.0,
        lambda_energy_cost=1.0)


@pytest.fixture(scope='function')
def exponential_reward():
    return ExpReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@pytest.fixture(scope='function')
def hourly_linear_reward():
    return HourlyLinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0))


@pytest.fixture(scope='function')
def normalized_linear_reward():
    return NormalizedLinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(
            20.0,
            23.5),
        range_comfort_summer=(
            23.0,
            26.0),
        summer_start=(6, 1),
        summer_final=(9, 30),
        energy_weight=0.5,
        max_energy_penalty=8,
        max_comfort_penalty=12,
    )


@pytest.fixture(scope='function')
def multizone_reward():
    return MultiZoneReward(
        energy_variables=['HVAC_electricity_demand_rate'],
        temperature_and_setpoints_conf={
            'air_temperature1': 'setpoint_temperature1',
            'air_temperature2': 'setpoint_temperature2'},
        comfort_threshold=0.5,
        energy_weight=0.5,
        lambda_energy=1.0,
        lambda_temperature=1.0)

# ---------------------------------------------------------------------------- #
#                         WHEN TESTS HAVE BEEN FINISHED                        #
# ---------------------------------------------------------------------------- #


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    # Deleting all temporal directories generated during tests (environments)
    directories = glob('PYTEST*/')
    for directory in directories:
        shutil.rmtree(directory)

    # Deleting all temporal directories generated during tests (simulators)
    directories = glob('PYTESTSIMULATOR*/')
    for directory in directories:
        shutil.rmtree(directory)

    # Deleting all temporal files generated during tests
    files = glob('./PYTEST*.xlsx')
    for file in files:
        os.remove(file)
    files = glob('./data_available*')
    for file in files:
        os.remove(file)

    # Deleting new JSON files generated during tests
    files = glob('sinergym/data/buildings/PYTEST*.epJSON')
    for file in files:
        os.remove(file)

    # Deleting new random weather files generated during tests
    files = glob('sinergym/data/weather/*Random*.epw')
    for file in files:
        os.remove(file)
