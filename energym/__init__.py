from gym.envs.registration import register

register(
    id='Eplus-demo-v1',
    entry_point='energym.envs:EplusDemo',
    kwargs = None
)