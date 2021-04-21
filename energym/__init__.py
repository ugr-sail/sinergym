from gym.envs.registration import register

register(
    id='Eplus-demo-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'variables_file': 'variables.cfg'
        'env_name' : 'demo-v1' 
    }
)

register(
    id='Eplus-discrete-hot-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variables.cfg'
        'env_name' : 'discrete-hot-v1' 
    }
)

register(
    id='Eplus-discrete-mixed-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variables.cfg',
        'env_name' : 'discrete-mixed-v1' 
    }
)

register(
    id='Eplus-discrete-cool-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variables.cfg',
        'env_name' : 'discrete-cool-v1' 
    }
)

register(
    id='Eplus-continuous-cool-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variables.cfg',
        'discrete_actions': False,
        'env_name' : 'continuous-cool-v1'
    }
)

register(
    id='Eplus-continuous-mixed-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variables.cfg',
        'discrete_actions': False,
        'env_name' : 'continuous-mixed-v1'
    }
)

register(
    id='Eplus-continuous-hot-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variables.cfg',
        'discrete_actions': False,
        'env_name' : 'continuous-hot-v1'
    }
)

register(
    id='Eplus-continuous-stochastic-hot-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variables.cfg',
        'discrete_actions': False,
        'weather_variability': (0.0, 2.5),
        'env_name' : 'continuous-stochastic-hot-v1'
    }
)

register(
    id='Eplus-discrete-stochastic-cool-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variables.cfg',
        'discrete_actions': True,
        'weather_variability': (0.0, 2.5),
        'env_name' : 'discrete-stochastic-cool-v1'
    }
)

register(
    id='Eplus-econimizer-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'discrete_actions': True,
        'env_name' : 'economizer-v1'
    }
)