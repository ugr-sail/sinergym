from gym.envs.registration import register

register(
    id='Eplus-discrete-v1',
    entry_point='energym.envs:EplusDemo',
)

register(
    id='Eplus-multi-v1',
    entry_point='energym.envs:EplusDemo',
    kwargs={'action_type': 'multidiscrete'}
)