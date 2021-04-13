import pytest
import energym.utils.common as common
from opyplus import Epm, WeatherData

@pytest.mark.parametrize(
	"st_year,st_mon,st_day,end_mon,end_day,expected",
	[
		(2000,10,1,11,1,2764800),
		(2002,1,10,2,5,2332800),
		(2021,5,5,5,5,3600*24),  # st_time=00:00:00 and ed_time=24:00:00
		(2004,7,1,6,1,-2505600), # Negative delta secons test
	]
)
def test_get_delta_seconds(st_year,st_mon,st_day,end_mon,end_day,expected):
	delta_sec=common.get_delta_seconds(st_year, st_mon, st_day, end_mon, end_day)
	assert type(delta_sec)==float
	assert delta_sec == expected


@pytest.fixture(scope="function")
def epm():
	idf_path="./energym/data/buildings/5ZoneAutoDXVAV.idf"
	return Epm.from_idf(idf_path)

@pytest.mark.parametrize(
	"sec_elapsed,expected_tuple",
	[
		(2764800,(2,2,0)),
		(0,(1,1,0)),
		((2764800*4)+(3600*10),(9,5,10)),
	]
)
def test_get_current_time_info(epm,sec_elapsed,expected_tuple):
	output = common.get_current_time_info(epm, sec_elapsed)
	assert type(output)==tuple
	assert len(output) == 3
	assert output == expected_tuple



def test_parse_variables():
	#The name of variables we expected
	expected=['Site Outdoor Air Drybulb Temperature', 'Site Outdoor Air Relative Humidity', 'Site Wind Speed',
			  'Site Wind Direction', 'Site Diffuse Solar Radiation Rate per Area', 'Site Direct Solar Radiation Rate per Area',
			  'Zone Thermostat Heating Setpoint Temperature', 'Zone Thermostat Cooling Setpoint Temperature',
			  'Zone Air Temperature', 'Zone Thermal Comfort Mean Radiant Temperature', 'Zone Air Relative Humidity',
			  'Zone Thermal Comfort Clothing Value', 'Zone Thermal Comfort Fanger Model PPD', 'Zone People Occupant Count',
			  'People Air Temperature', 'Facility Total HVAC Electric Demand Power']

	var_file="./energym/data/variables/variables.cfg"
	variables=common.parse_variables(var_file)

	assert type(variables)==list
	assert len(variables)== 16
	for i in range(len(variables)):
		assert variables[i]==expected[i]

@pytest.fixture(scope="session")
def weather_file():
	return './energym/data/weather/USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw'

@pytest.fixture(scope="function")
def weather_data(weather_file):
	return WeatherData.from_epw(weather_file)


@pytest.mark.parametrize(
	"variation",
	[
		(None),
		((1,0.2)),
		((5,0.5)),
		((0,0)),
	]
)
def test_create_variable_weather(variation,weather_data,weather_file):
	output = common.create_variable_weather(weather_data,weather_file,['drybulb'],variation)
	if variation is None:
		assert output is None
	else:
		expected = weather_file.split('.epw')[0]+'_Random_'+str(variation[0])+'_'+str(variation[1])+'.epw'
		assert output==expected




