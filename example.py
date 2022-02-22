import xml.etree.ElementTree as ET
from xml.dom import minidom

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
        'Facility Total HVAC Electricity Demand Rate (Whole Building)'],
    'action_variables': [
        'West-HtgSetP-RL',
        'West-ClgSetP-RL',
        'East-HtgSetP-RL',
        'East-ClgSetP-RL']}

config_params = {}

variables_custom = ET.Element('BCVTB-variables')

variables_custom.append(ET.Comment('Receivedjeje from EnergyPlus'))

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

else:
    # Copy default variables.cfg
    tree = ET.parse('./sinergym/data/variables/variablesDXVAV.cfg')
    default_variables = tree.getroot()
    for var in default_variables.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            variables_custom.append(var)

xmlstr = minidom.parseString(
    ET.tostring(variables_custom)).toprettyxml(
        indent="   ")
with open("prueba.cfg", "w") as f:
    f.write(xmlstr)

# arbol = ET.ElementTree(variables_custom)
# arbol.write("./prueba.xml")
