import xml.etree.ElementTree as ET

config_params = {
    'observation_variables': [
        'Site Outdoor Air Drybulb Temperature (Environment)',
        'Site Outdoor Air Relative Humidity (Environment)',
        'Site Wind Speed (Environment)',
        'Site Wind Direction (Environment)',
        'Site Diffuse Solar Radiation Rate per Area (Environment)',
        'Site Direct Solar Radiation Rate per Area (Environment)',
        'Zone Thermostat Heating Setpoint Temperature (SPACE1-1)',
        'Zone Thermostat Cooling Setpoint Temperature (SPACE1-1)',
        'Zone Air Temperature (SPACE1-1)',
        'Zone Thermal Comfort Mean Radiant Temperature (SPACE1-1 PEOPLE 1)',
        'Zone Air Relative Humidity (SPACE1-1)',
        'Zone Thermal Comfort Clothing Value (SPACE1-1 PEOPLE 1)',
        'Zone Thermal Comfort Fanger Model PPD (SPACE1-1 PEOPLE 1)',
        'Zone People Occupant Count (SPACE1-1)',
        'People Air Temperature (SPACE1-1 PEOPLE 1)',
        'Facility Total HVAC Electricity Demand Rate (Whole Building)']}

variables_custom = ET.Element('BCVTB-variables')

if 'observation_variables' in config_params:
    for variable in config_params['observation_variables']:
        # variable = "<variable_name> (<variable_zone>)"
        variable_elements = variable.split('(')
        variable_name = variable_elements[0]
        variable_zone = variable_elements[1][:-1]

        new_xml_variable = ET.SubElement(
            variables_custom, 'variable', source='EnergyPlus')
        ET.SubElement(
            new_xml_variable,
            'EnergyPlus',
            name=variable_zone,
            type=variable_name)

arbol = ET.ElementTree(variables_custom)
arbol.write("./prueba.xml")
