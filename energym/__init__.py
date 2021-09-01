from gym.envs.registration import register
from energym.utils.rewards import LinearReward
#========================5ZoneAutoDXVAV========================#
register(
    id='Eplus-demo-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'reward': LinearReward(),
        'env_name': 'demo-v1'
    }
)

register(
    id='Eplus-5Zone-hot-discrete-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward(),
        'env_name': '5Zone-hot-discrete-v1'
    }
)

register(
    id='Eplus-5Zone-mixed-discrete-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward(),
        'env_name': '5Zone-mixed-discrete-v1'
    }
)

register(
    id='Eplus-5Zone-cool-discrete-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward(),
        'env_name': '5Zone-cool-discrete-v1'
    }
)

register(
    id='Eplus-5Zone-hot-discrete-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': '5Zone-hot-discrete-stochastic-v1'
    }
)

register(
    id='Eplus-5Zone-mixed-discrete-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': '5Zone-mixed-discrete-stochastic-v1'
    }
)

register(
    id='Eplus-5Zone-cool-discrete-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': '5Zone-cool-discrete-stochastic-v1'
    }
)

register(
    id='Eplus-5Zone-hot-continuous-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward(),
        'env_name': '5Zone-hot-continuous-v1'
    }
)

register(
    id='Eplus-5Zone-mixed-continuous-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward(),
        'env_name': '5Zone-mixed-continuous-v1'
    }
)

register(
    id='Eplus-5Zone-cool-continuous-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward(),
        'env_name': '5Zone-cool-continuous-v1'
    }
)

register(
    id='Eplus-5Zone-hot-continuous-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': '5Zone-hot-continuous-stochastic-v1'
    }
)

register(
    id='Eplus-5Zone-mixed-continuous-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': '5Zone-mixed-continuous-stochastic-v1'
    }
)

register(
    id='Eplus-5Zone-cool-continuous-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': '5Zone-cool-continuous-stochastic-v1'
    }
)

#========================DATACENTER========================#

register(
    id='Eplus-datacenter-discrete-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward(),
        'env_name': 'datacenter-discrete-v1'
    }
)

register(
    id='Eplus-datacenter-continuous-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward(),
        'env_name': 'datacenter-continuous-v1'
    }
)

register(
    id='Eplus-datacenter-discrete-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': 'datacenter-discrete-stochastic-v1'
    }
)

register(
    id='Eplus-datacenter-continuous-stochastic-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward(),
        'env_name': 'datacenter-continuous-stochastic-v1'
    }
)

#========================MULLION========================#

register(
    id='Eplus-IWMullion-discrete-v1',
    entry_point='energym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward(),
        'env_name': 'IWMullion-discrete-v1'
    }
)
