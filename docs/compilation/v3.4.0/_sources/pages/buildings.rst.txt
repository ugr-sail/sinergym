############
Buildings
############

This section details the buildings incorporated in Sinergym, separate from the other 
components that constitute the environment for each experiment. It also provides 
the default *Sinergym* observation and action spaces for each building, though 
these spaces can be customized by the user. The observation variables names would
be the base name with each key at the begginning.

.. important:: To add new buildings to this framework, please refer to the section 
               :ref:`Adding new buildings for environments`.

**************************
Datacenter
**************************

**2ZoneDataCenterHVAC_wEconomizer.epJSON:**
A 491.3 m2 building divided into two asymmetrical zones (west and east zone). 
Each zone has an HVAC system consisting of air economizers, evaporative coolers, 
DX cooling coil, chilled water coil, and VAV units. The main source of heat 
comes from the hosted servers. The heating and cooling setpoint values are 
applied simultaneously to both zones.

.. image:: /_static/datacenter.png
  :width: 700
  :alt: Datacenter building
  :align: center

The default actuators, output and meters variables are:

+-------------------+---------------------+-----------------+-----------------+
| Actuator          | Variable Name       | Element Type    | Value Type      |
+===================+=====================+=================+=================+
| Heating Setpoints | Heating_Setpoint_RL | Schedule:Compact| Schedule Value  |
+-------------------+---------------------+-----------------+-----------------+
| Cooling Setpoints | Cooling_Setpoint_RL | Schedule:Compact| Schedule Value  |
+-------------------+---------------------+-----------------+-----------------+

+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Variable                                      | Base Variable Name                            | Keys                              |
+===============================================+===============================================+===================================+
| Site Outdoor Air Drybulb Temperature          | outdoor_temperature                           | Environment                       |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Site Outdoor Air Relative Humidity            | outdoor_humidity                              | Environment                       |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Site Wind Speed                               | wind_speed                                    | Environment                       |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Site Wind Direction                           | wind_direction                                | Environment                       |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Site Diffuse Solar Radiation Rate per Area    | diffuse_solar_radiation                       | Environment                       |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Site Direct Solar Radiation Rate per Area     | direct_solar_radiation                        | Environment                       |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Thermostat Heating Setpoint Temperature  | htg_setpoint                                  | West Zone, East Zone              |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Thermostat Cooling Setpoint Temperature  | clg_setpoint                                  | West Zone, East Zone              |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Air Temperature                          | air_temperature                               | West Zone, East Zone              |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Air Relative Humidity                    | air_humidity                                  | West Zone, East Zone              |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Thermal Comfort Mean Radiant Temperature | thermal_comfort_mean_radiant_temperature      | West Zone PEOPLE, East Zone PEOPLE|
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Thermal Comfort Clothing Value           | thermal_comfort_clothing_value                | West Zone PEOPLE, East Zone PEOPLE|
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone Thermal Comfort Fanger Model PPD         | thermal_comfort_fanger_model_ppd              | West Zone PEOPLE, East Zone PEOPLE|
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Zone People Occupant Count                    | people_occupant                               | West Zone, East Zone              |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| People Air Temperature                        | people_air_temperature                        | West Zone PEOPLE, East Zone PEOPLE|
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+
| Facility Total HVAC Electricity Demand Rate   | HVAC_electricity_demand_rate                  | Whole Building                    |
+-----------------------------------------------+-----------------------------------------------+-----------------------------------+

.. warning:: Since the update to EnergyPlus version 23.1.0, it appears that temperature setpoints are not correctly 
             applied in the East zone. The issue is currently under investigation. In the meantime, the default 
             reward functions only apply to the control of the West zone to maintain result consistency. For more
             information about this issue, visit `#395 <https://github.com/ugr-sail/sinergym/issues/395>`__.

**************************
Small Datacenter
**************************

**1ZoneDataCenterCRAC_wApproachTemp.epJSON**:
This file demonstrates a simple data center model with air-cooled IT equipment 
(ITE) served by a CRAC system. The air-cooled ITE illustrates the user of various schedules 
and curves to vary server power use. The CRAC system has been setup to represent a 
Lieber DSE 125 with pumped refrigerant economizer DX cooling coil system.Fictional 
1 zone building with resistive walls. No windows. Data Center server ITE object 
for internal gains.  No internal mass.  The building is oriented due north.


.. image:: /_static/small_datacenter.png
  :width: 700
  :alt: Small Datacenter building
  :align: center

The default actuators, output and meters variables are:

+----------------------------------+---------------------+-----------------+-----------------+
| Actuator                         | Variable Name       | Element Type    | Value Type      |
+==================================+=====================+=================+=================+
| COOLING RETURN AIR SETPOINT      | Cooling_Setpoint_RL | Schedule:Compact| Schedule Value  |
| SCHEDULE                         |                     |                 |                 |
+----------------------------------+---------------------+-----------------+-----------------+
| SUPPLY AIR SETPOINT SCHEDULE     | Supply_Air_RL       | Schedule:Compact| Schedule Value  |
+----------------------------------+---------------------+-----------------+-----------------+

+------------------------------------------------+----------------------------------+---------------------+
| Variable                                       | Base Variable Name               | Key                 |
+================================================+==================================+=====================+
| Site Outdoor Air Drybulb Temperature           | outdoor_temperature              | Environment         |
+------------------------------------------------+----------------------------------+---------------------+
| Site Outdoor Air Relative Humidity             | outdoor_humidity                 | Environment         |
+------------------------------------------------+----------------------------------+---------------------+
| Site Wind Speed                                | wind_speed                       | Environment         |
+------------------------------------------------+----------------------------------+---------------------+
| Site Wind Direction                            | wind_direction                   | Environment         |
+------------------------------------------------+----------------------------------+---------------------+
| Site Diffuse Solar Radiation Rate per Area     | diffuse_solar_radiation          | Environment         |
+------------------------------------------------+----------------------------------+---------------------+
| Site Direct Solar Radiation Rate per Area      | direct_solar_radiation           | Environment         |
+------------------------------------------------+----------------------------------+---------------------+
| Zone Thermostat Heating Setpoint Temperature   | htg_setpoint                     | Main Zone           |
+------------------------------------------------+----------------------------------+---------------------+
| Zone Thermostat Cooling Setpoint Temperature   | clg_setpoint                     | Main Zone           |
+------------------------------------------------+----------------------------------+---------------------+
| Zone Air Temperature                           | air_temperature                  | Main Zone           |
+------------------------------------------------+----------------------------------+---------------------+
| Zone Air Relative Humidity                     | air_humidity                     | Main Zone           |
+------------------------------------------------+----------------------------------+---------------------+
| Cooling Coil Electricity Rate                  | cooling_coil_demand_rate         | MAIN COOLING COIL 1 |
+------------------------------------------------+----------------------------------+---------------------+
| Fan Electricity Rate                           | fan_demand_rate                  | EC PLUG FAN 1       |
+------------------------------------------------+----------------------------------+---------------------+
| ITE UPS Electricity Rate                       | ups_demand_rate                  | DATA CENTER SERVERS |
+------------------------------------------------+----------------------------------+---------------------+
| ITE Fan Electricity Rate                       | ite_fan_demand_rate              | DATA CENTER SERVERS |
+------------------------------------------------+----------------------------------+---------------------+
| ITE CPU Electricity Rate                       | cpu_demand_rate                  | DATA CENTER SERVERS |
+------------------------------------------------+----------------------------------+---------------------+
| Facility Total HVAC Electricity Demand Rate    | HVAC_electricity_demand_rate     | Whole Building      |
+------------------------------------------------+----------------------------------+---------------------+
| Facility Total Building Electricity Demand Rate| building_electricity_demand_rate | Whole Building      |
+------------------------------------------------+----------------------------------+---------------------+
| Facility Total Electricity Demand Rate         | total_electricity_demand_rate    | Whole Building      |
+------------------------------------------------+----------------------------------+---------------------+

**************************
5Zone
**************************

**5ZoneAutoDXVAV.epJSON:**
A single-story building divided
into 5 zones (1 indoor and 4 outdoor). Its surface area is 463.6
m2, and it is equipped with a VAV package (DX cooling coil
and gas heating coils) with fully auto-sized input as the HVAC
system to be controlled.

.. image:: /_static/5zone.png
  :width: 700
  :alt: 5Zone building
  :align: center

The default actuators, output and meters variables are:

+----------------+---------------------+-----------------+-----------------+
| Actuator       | Variable Name       | Element Type    | Value Type      |
+================+=====================+=================+=================+
| HTG-SETP-SCH   | Heating_Setpoint_RL | Schedule:Compact| Schedule Value  |
+----------------+---------------------+-----------------+-----------------+
| CLG-SETP-SCH   | Cooling_Setpoint_RL | Schedule:Compact| Schedule Value  |
+----------------+---------------------+-----------------+-----------------+

+------------------------------------------------+----------------------------------+-----------------+
| Variable                                       | Base Variable Name               | Key             |
+================================================+==================================+=================+
| Site Outdoor Air DryBulb Temperature           | outdoor_temperature              | Environment     |
+------------------------------------------------+----------------------------------+-----------------+
| Site Outdoor Air Relative Humidity             | outdoor_humidity                 | Environment     |
+------------------------------------------------+----------------------------------+-----------------+
| Site Wind Speed                                | wind_speed                       | Environment     |
+------------------------------------------------+----------------------------------+-----------------+
| Site Wind Direction                            | wind_direction                   | Environment     |
+------------------------------------------------+----------------------------------+-----------------+
| Site Diffuse Solar Radiation Rate per Area     | diffuse_solar_radiation          | Environment     |
+------------------------------------------------+----------------------------------+-----------------+
| Site Direct Solar Radiation Rate per Area      | direct_solar_radiation           | Environment     |
+------------------------------------------------+----------------------------------+-----------------+
| Zone Thermostat Heating Setpoint Temperature   | htg_setpoint                     | SPACE5-1        |
+------------------------------------------------+----------------------------------+-----------------+
| Zone Thermostat Cooling Setpoint Temperature   | clg_setpoint                     | SPACE5-1        |
+------------------------------------------------+----------------------------------+-----------------+
| Zone Air Temperature                           | air_temperature                  | SPACE5-1        |
+------------------------------------------------+----------------------------------+-----------------+
| Zone Air Relative Humidity                     | air_humidity                     | SPACE5-1        |
+------------------------------------------------+----------------------------------+-----------------+
| Zone People Occupant Count                     | people_occupant                  | SPACE5-1        |
+------------------------------------------------+----------------------------------+-----------------+
| Environmental Impact Total CO2 Emissions Carbon| co2_emission                     | site            |
| Equivalent Mass                                |                                  |                 |
+------------------------------------------------+----------------------------------+-----------------+
| Facility Total HVAC Electricity Demand Rate    | HVAC_electricity_demand_rate     | Whole Building  |
+------------------------------------------------+----------------------------------+-----------------+

+------------------+------------------------+
| Meter            | Variable Name          |
+==================+========================+
| Electricity:HVAC | total_electricity_HVAC |
+------------------+------------------------+

**************************
Warehouse
**************************

**ASHRAE9012016_Warehouse.epJSON:**
It is a non-residential 4598 m2 floor building, 
divided into 3 zones: bulk storage, fine storage and an office. 
The Office zone is enclosed on two sides and at the top by the 
Fine Storage zone, and it is the unique zone with windows. 
Available fuel types are gas and electricity, and it is equipped 
with HVAC system.

.. image:: /_static/warehouse.png
  :width: 700
  :alt: Warehouse building
  :align: center

The default actuators, output and meters variables are:

+------------------------+-------------------+----------------+-----------------+
| Actuator               | Variable Name     | Element Type   | Value Type      |
+========================+===================+================+=================+
| Office Heating Schedule| Office_Heating_RL | Schedule:Year  | Schedule Value  |
+------------------------+-------------------+----------------+-----------------+
| Office Cooling Schedule| Office_Cooling_RL | Schedule:Year  | Schedule Value  |
+------------------------+-------------------+----------------+-----------------+

+------------------------------------------------+----------------------------------+-----------------------------------+
| Variable                                       | Base Variable Name               | Key                               |
+================================================+==================================+===================================+
| Site Outdoor Air DryBulb Temperature           | outdoor_temperature              | Environment                       |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Site Outdoor Air Relative Humidity             | outdoor_humidity                 | Environment                       |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Site Wind Speed                                | wind_speed                       | Environment                       |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Site Wind Direction                            | wind_direction                   | Environment                       |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Site Diffuse Solar Radiation Rate per Area     | diffuse_solar_radiation          | Environment                       |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Site Direct Solar Radiation Rate per Area      | direct_solar_radiation           | Environment                       |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Zone Thermostat Heating Setpoint Temperature   | htg_setpoint                     | Zone1 Office, Zone2 Fine Storage, |
|                                                |                                  | Zone3 Bulk Storage                |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Zone Thermostat Cooling Setpoint Temperature   | clg_setpoint                     | Zone1 Office, Zone2 Fine Storage  |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Zone Air Temperature                           | air_temperature                  | Zone1 Office, Zone2 Fine Storage, |
|                                                |                                  | Zone3 Bulk Storage                |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Zone Air Relative Humidity                     | air_humidity                     | Zone1 Office, Zone2 Fine Storage, |
|                                                |                                  | Zone3 Bulk Storage                |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Zone People Occupant Count                     | people_occupant                  | Zone1 Office                      |
+------------------------------------------------+----------------------------------+-----------------------------------+
| Facility Total HVAC Electricity Demand Rate    | HVAC_electricity_demand_rate     | Whole Building                    |
+------------------------------------------------+----------------------------------+-----------------------------------+

**************************
OfficeMedium
**************************

**ASHRAE9012016_OfficeMedium.epJSON:**
It is a 4979.6 m2 building with 3 floors. Each floor has 
four perimeter zones and one core zone. Available fuel types 
are gas and electricity, and it is equipped with HVAC system.

.. image:: /_static/officeMedium.png
  :width: 700
  :alt: OfficeMedium building
  :align: center

The default actuators, output and meters variables are:

+------------------------+-------------------+-----------------+-----------------+
| Actuator               | Variable Name     | Element Type    | Value Type      |
+========================+===================+=================+=================+
| HTGSETP_SCH_YES_OPTIMUM| Office_Heating_RL | Schedule:Compact| Schedule Value  |
+------------------------+-------------------+-----------------+-----------------+
| CLGSETP_SCH_YES_OPTIMUM| Office_Cooling_RL | Schedule:Compact| Schedule Value  |
+------------------------+-------------------+-----------------+-----------------+

+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Variable                                       | Base Variable Name               | Key                                                                             |
+================================================+==================================+=================================================================================+
| Site Outdoor Air DryBulb Temperature           | outdoor_temperature              | Environment                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Site Outdoor Air Relative Humidity             | outdoor_humidity                 | Environment                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Site Wind Speed                                | wind_speed                       | Environment                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Site Wind Direction                            | wind_direction                   | Environment                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Site Diffuse Solar Radiation Rate per Area     | diffuse_solar_radiation          | Environment                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Site Direct Solar Radiation Rate per Area      | direct_solar_radiation           | Environment                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Zone Thermostat Heating Setpoint Temperature   | htg_setpoint                     | Core_bottom                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Zone Thermostat Cooling Setpoint Temperature   | clg_setpoint                     | Core_bottom                                                                     |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Zone Air Temperature                           | air_temperature                  | Core_bottom, Core_mid, Core_top,                                                |
|                                                |                                  | FirstFloor_Plenum, MidFloor_Plenum, TopFloor_Plenum,                            |
|                                                |                                  | Perimeter_bot_ZN_1, Perimeter_bot_ZN_2, Perimeter_bot_ZN_3, Perimeter_bot_ZN_4, |
|                                                |                                  | Perimeter_mid_ZN_1, Perimeter_mid_ZN_2, Perimeter_mid_ZN_3, Perimeter_mid_ZN_4, |
|                                                |                                  | Perimeter_top_ZN_1, Perimeter_top_ZN_2, Perimeter_top_ZN_3, Perimeter_top_ZN_4  |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Zone Air Relative Humidity                     | air_humidity                     | Core_bottom, Core_mid, Core_top,                                                |
|                                                |                                  | FirstFloor_Plenum, MidFloor_Plenum, TopFloor_Plenum,                            |
|                                                |                                  | Perimeter_bot_ZN_1, Perimeter_bot_ZN_2, Perimeter_bot_ZN_3, Perimeter_bot_ZN_4, |
|                                                |                                  | Perimeter_mid_ZN_1, Perimeter_mid_ZN_2, Perimeter_mid_ZN_3, Perimeter_mid_ZN_4, |
|                                                |                                  | Perimeter_top_ZN_1, Perimeter_top_ZN_2, Perimeter_top_ZN_3, Perimeter_top_ZN_4  |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+
| Facility Total HVAC Electricity Demand Rate    | HVAC_electricity_demand_rate     | Whole Building                                                                  |
+------------------------------------------------+----------------------------------+---------------------------------------------------------------------------------+

**************************
ShopWithVanBattery
**************************

**ShopWithVanBattery.epJSON:**
It is a 390.2 m2 building, with only one floor. It has five
zones; four of them are perimeter zones and one in the center.
This is a low-energy building with photovoltaic panel and 
electrical storage. It demonstrates the use of the battery 
model for electrical storage. It has a full HVAC model and 
water heating service.
This is a small repair shop. Open Monday through Friday,
45 hours per week. 

.. image:: /_static/shop.png
  :width: 700
  :alt: Shop building
  :align: center

The default actuators, output and meters variables are:

+--------------+---------------------+-----------------+-----------------+
| Actuator     | Variable Name       | Element Type    | Value Type      |
+==============+=====================+=================+=================+
| HTGSETP_SCH  | Heating_Setpoint_RL | Schedule:Compact| Schedule Value  |
+--------------+---------------------+-----------------+-----------------+
| CLGSETP_SCH  | Cooling_Setpoint_RL | Schedule:Compact| Schedule Value  |
+--------------+---------------------+-----------------+-----------------+

+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Variable                                       | Base Variable Name          | Key                                                               |
+================================================+=============================+===================================================================+
| Site Outdoor Air Drybulb Temperature           | outdoor_temperature         | Environment                                                       |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Site Outdoor Air Relative Humidity             | outdoor_humidity            | Environment                                                       |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Site Wind Speed                                | wind_speed                  | Environment                                                       |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Site Wind Direction                            | wind_direction              | Environment                                                       |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Site Diffuse Solar Radiation Rate per Area     | diffuse_solar_radiation     | Environment                                                       |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Site Direct Solar Radiation Rate per Area      | direct_solar_radiation      | Environment                                                       |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Zone Thermostat Heating Setpoint Temperature   | htg_setpoint                | ZN_1_FLR_1_SEC_5                                                  |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Zone Thermostat Cooling Setpoint Temperature   | clg_setpoint                | ZN_1_FLR_1_SEC_5                                                  |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Battery Charge State          | storage_battery_charge_state| Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Charge Energy                 | storage_charge_energy       | Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Charge Power                  | storage_charge_power        | Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Discharge Energy              | storage_discharge_energy    | Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Discharge Power               | storage_discharge_power     | Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Thermal Loss Energy           | storage_thermal_loss_energy | Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Electric Storage Thermal Loss Rate             | storage_thermal_loss_rate   | Kibam                                                             |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Zone Air Temperature                           | air_temperature             | ZN_1_FLR_1_SEC_1, ZN_1_FLR_1_SEC_2, ZN_1_FLR_1_SEC_3,             |
|                                                |                             | ZN_1_FLR_1_SEC_4, ZN_1_FLR_1_SEC_5                                |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Zone Air Relative Humidity                     | air_humidity                | ZN_1_FLR_1_SEC_1, ZN_1_FLR_1_SEC_2, ZN_1_FLR_1_SEC_3,             |
|                                                |                             | ZN_1_FLR_1_SEC_4, ZN_1_FLR_1_SEC_5                                |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Zone People Occupant Count                     | people_count                | ZN_1_FLR_1_SEC_1, ZN_1_FLR_1_SEC_2, ZN_1_FLR_1_SEC_3,             |
|                                                |                             | ZN_1_FLR_1_SEC_4, ZN_1_FLR_1_SEC_5                                |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+
| Facility Total HVAC Electricity Demand Rate    | HVAC_electricity_demand_rate| Whole Building                                                    |
+------------------------------------------------+-----------------------------+-------------------------------------------------------------------+

**************************
OfficeGridStorageSmoothing
**************************

**OfficeGridStorageSmoothing.epJSON:**
It is a large office building with 12 floors and a basement, with a rectangular aspect and 46,320 m2. 
The building is organized in 19 zones: the basement, bot, mid and top level. Each level has a
core zone and 4 perimeter zones. Floor zone is described for bot, mid and top level too.
It has a battery control for charging and discharging from the grid.

.. image:: /_static/officeGrid.png
  :width: 700
  :alt: Shop building
  :align: center

The default actuators, output and meters variables are:

+-------------------+---------------------+-----------------+-----------------+
| Actuator          | Variable Name       | Element Type    | Value Type      |
+===================+=====================+=================+=================+
| HTGSETP_SCH       | Heating_Setpoint_RL | Schedule:Compact| Schedule Value  |
+-------------------+---------------------+-----------------+-----------------+
| CLGSETP_SCH       | Cooling_Setpoint_RL | Schedule:Compact| Schedule Value  |
+-------------------+---------------------+-----------------+-----------------+
| Charge Schedule   | Charge_Rate_RL      | Schedule:Compact| Schedule Value  |
+-------------------+---------------------+-----------------+-----------------+
| Discharge Schedule| Discharge_Rate_RL   | Schedule:Compact| Schedule Value  |
+-------------------+---------------------+-----------------+-----------------+

+-----------------------------------------------+-----------------------------+-----------------+
| Variable                                      | Base Variable Name          | Keys            |
+===============================================+=============================+=================+
| Site Outdoor Air DryBulb Temperature          | outdoor_temperature         | Environment     |
+-----------------------------------------------+-----------------------------+-----------------+
| Site Outdoor Air Relative Humidity            | outdoor_humidity            | Environment     |
+-----------------------------------------------+-----------------------------+-----------------+
| Site Wind Speed                               | wind_speed                  | Environment     |
+-----------------------------------------------+-----------------------------+-----------------+
| Site Wind Direction                           | wind_direction              | Environment     |
+-----------------------------------------------+-----------------------------+-----------------+
| Site Diffuse Solar Radiation Rate per Area    | diffuse_solar_radiation     | Environment     |
+-----------------------------------------------+-----------------------------+-----------------+
| Site Direct Solar Radiation Rate per Area     | direct_solar_radiation      | Environment     |
+-----------------------------------------------+-----------------------------+-----------------+
| Zone Thermostat Heating Setpoint Temperature  | htg_setpoint                | Basement        |
+-----------------------------------------------+-----------------------------+-----------------+
| Zone Thermostat Cooling Setpoint Temperature  | clg_setpoint                | Basement        |
+-----------------------------------------------+-----------------------------+-----------------+
| Zone Air Temperature                          | air_temperature             | Multiple Keys   |
+-----------------------------------------------+-----------------------------+-----------------+
| Zone Air Relative Humidity                    | air_humidity                | Multiple Keys   |
+-----------------------------------------------+-----------------------------+-----------------+
| Zone People Occupant Count                    | people_count                | Multiple Keys   |
+-----------------------------------------------+-----------------------------+-----------------+
| Electric Storage Simple Charge State          | battery_charge_state        | Battery         |
+-----------------------------------------------+-----------------------------+-----------------+
| Facility Total HVAC Electricity Demand Rate   | HVAC_electricity_demand_rate| Whole Building  |
+-----------------------------------------------+-----------------------------+-----------------+