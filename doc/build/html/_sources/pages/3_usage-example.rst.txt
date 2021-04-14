#############
Usage example
#############

Energym uses the standard OpenAI gym API. So basic loop should be
something like:

.. code:: python

    import gym
    import energym

    env = gym.make('Eplus-demo-v1')
    obs = env.reset()
    done = False
    R = 0.0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        R += reward
    print('Total reward for the episode: %.4f' % R)
    env.close()

Notice that a folder will be created in the working directory after
creating the environment. It will contain the EnergyPlus outputs
produced during the simulation.