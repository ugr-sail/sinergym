import gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import LoggerWrapper
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from opyplus import Epm, Idd, WeatherData

extra_conf = {
    'action_definition': {
        'ThermostatSetpoint:DualSetpoint': [{
            'name': 'space1-dualsetp-rl',
            'heating_name': 'space1-htgsetp-rl',
            'cooling_name': 'space1-clgsetp-rl',
            'zones': ['space1-1']
        }]
    }
}

_idf_path = './sinergym/data/buildings/5ZoneAutoDXVAV.idf'
variables_tree = ET.Element('BCVTB-variables')
_idd = Idd(os.path.join(os.environ['EPLUS_PATH'], 'Energy+.idd'))
building = Epm.from_idf(
    _idf_path,
    idd_or_version=_idd,
    check_length=False)

if extra_conf.get('action_definition'):
    action_definition = extra_conf['action_definition']
    for controller_type, controllers in action_definition.items():
        # ThermostatSetpoint:DualSetpoint IDF Management
        if controller_type == 'ThermostatSetpoint:DualSetpoint':
            for controller in controllers:
                # Create Ptolomy variables
                building.ExternalInterface_Schedule.add(
                    name=controller['heating_name'],
                    schedule_type_limits_name='Temperature',
                    initial_value=21)
                building.ExternalInterface_Schedule.add(
                    name=controller['cooling_name'],
                    schedule_type_limits_name='Temperature',
                    initial_value=25)
                # Create a ThermostatSetpoint:DualSetpoint object
                building.ThermostatSetpoint_DualSetpoint.add(
                    name=controller['name'],
                    heating_setpoint_temperature_schedule_name=controller['heating_name'],
                    cooling_setpoint_temperature_schedule_name=controller['cooling_name'])
                # Link in zones required
                for zone_control in building.ZoneControl_Thermostat:
                    if zone_control.zone_or_zonelist_name.name in controller['zones']:
                        zone_control.control_3_name = controller['name']


episode_idf_path = './prueba.idf'
building.save(episode_idf_path)
