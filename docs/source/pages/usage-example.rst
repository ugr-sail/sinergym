#############
Usage example
#############

If you used our Dockerfile during installation, you should have the *try_env.py* file in your workspace as soon as you enter in. In case you have installed everything on your local machine directly, place it inside our cloned repository.
In any case, we start from the point that you have at your disposal a terminal with the appropriate python version and Sinergym running correctly.

At this point of the documentation, we have explained how to install Sinergym, the environments it includes, the reward functions we can define, wrappers and controllers.

In this section we will see some examples of use to better understand how they can be used in practice.

*****************
Simplest example
*****************

Let's start with the simplest use case for the Sinergym tool. In the root repository we have the script **try_env.py**:

.. literalinclude:: ../../../try_env.py
    :language: python

The **Sinergym import** is really important, because without it the ID's of our environments will not have been registered in the gym module and therefore we cannot use our buildings as gym environments.

We create our env with **gym.make** and we run the simulation for one episode (`for i in range(1)`). We collect the rewards returned by the environment and calculate their average each month of simulation.

The action taken at each step is randomly chosen from its action space defined under the gym standard. When we have finished displaying the results on the screen and the episode is finished, we close the environment with `env.close()`.

.. note:: We will use this simple example as a basis and will add new elements in the following examples in this section.


*****************
Adding a reward
*****************

By default, all our environment ID's make use of a default (linear) reward. But this reward can be changed by adding this parameter to the constructor of our environment:

.. code:: python

    import gym
    import numpy as np

    import sinergym
    from sinergym.utils.rewards import LinearReward, ExpReward

    env = gym.make('Eplus-5Zone-hot-continuous-v1', reward=ExpReward, reward_kwargs={
                                                                        'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                                                        'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                                                        'range_comfort_winter': (20.0, 23.5),
                                                                        'range_comfort_summer': (23.0, 26.0),
                                                                        'energy_weight': 0.1})
    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
        while not done:
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:  # display results every month
                current_month = info['month']
                print('Reward: ', sum(rewards), info)
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    env.close()

.. warning:: When specifying a different reward with `gym.make` than the default environment ID, it is very important to set the `reward_kwargs` that are required and therefore do not have a default value. In the rewards we have defined it is required: **temperature_variable(s)**, **energy_variable**, **range_comfort_winter**, **range_comfort_summer**. The rest of them have default values and it is not neccesary to specify.

This example is exactly the same as the previous one, except that it uses different criteria to determine the rewards in each step of the simulation. 
If you run the code you can see the difference in the values obtained for the reward (using a seed for randomization).


*****************
Adding wrappers
*****************

By default, the ID's of our environments do not include any wrapper, but we can add them after the creation of the environment:

.. code:: python

    import gym
    import numpy as np

    import sinergym
    from sinergym.utils.rewards import LinearReward, ExpReward
    from sinergym.utils.wrapper import LoggerWrapper, NormalizeObservation

    env = gym.make('Eplus-5Zone-hot-continuous-v1', reward=ExpReward, reward_kwargs={
                                                                        'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                                                        'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                                                        'range_comfort_winter': (20.0, 23.5),
                                                                        'range_comfort_summer': (23.0, 26.0),
                                                                        'energy_weight': 0.1})
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)
    ...

    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
        while not done:
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:  # display results every month
                current_month = info['month']
                print('Reward: ', sum(rewards), info)
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    env.close()

With this, we have added normalization to the observations returned by the environment and Sinergym will also store the outputs in a CSV. 
For more information about how Sinergym displays its output, please visit the section :ref:`Output format`.

******************************
Using a rule-based controller
******************************

You can replace the random actions we have used in the previous examples with one of our rule-based controllers for that type of environment (5Zone IDF):

.. code:: python

    import gym
    import numpy as np

    import sinergym
    from sinergym.utils.rewards import LinearReward, ExpReward
    from sinergym.utils.wrapper import LoggerWrapper, NormalizeObservation
    from sinergym.utils.controllers import RBC5Zone

    env = gym.make('Eplus-5Zone-hot-continuous-v1', reward=ExpReward , reward_kwargs={
                                                                        'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                                                        'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                                                        'range_comfort_winter': (20.0, 23.5),
                                                                        'range_comfort_summer': (23.0, 26.0),
                                                                        'energy_weight': 0.1})
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)

    agent = RBC5Zone(env)

    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
        while not done:
            a = agent.act(obs)
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:  # display results every month
                current_month = info['month']
                print('Reward: ', sum(rewards), info)
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    env.close()

.. note:: You can also use our rule-based controller for Datacenter called **RBCDatacenter** if the environment is of that type or a random agent called **RandomController** in every environment.

******************************************************
Overwriting some default values of the environments.
******************************************************

In the same way that we can change the default reward function, as we have done in the second example, 
it is possible to substitute other default values of the environment ID. 

You can change the weather file, the number of timesteps an action repeats (default 1), 
the last n episodes you want to be stored in the Sinergym output folder (default 10), 
the name of the environment or the variability in stochastic environments:

.. code:: python

    import gym
    import numpy as np

    import sinergym
    from sinergym.utils.rewards import LinearReward, ExpReward
    from sinergym.utils.wrapper import LoggerWrapper, NormalizeObservation
    from sinergym.utils.controllers import   RBCDatacenter

    env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1', 
                    reward=ExpReward, reward_kwargs={
                                        'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                        'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                        'range_comfort_winter': (20.0, 23.5),
                                        'range_comfort_summer': (23.0, 26.0),
                                        'energy_weight': 0.1},
                    weather_file='ESP_Granada.084190_SWEC.epw',
                    weather_variability=(1.0,0.0,0.001),
                    env_name='new_env_name',
                    act_repeat=4,
                    max_ep_data_store_num = 20)

    env = NormalizeObservation(env)
    env = LoggerWrapper(env)

    agent = RBCDatacenter(env)

    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
        while not done:
            a = agent.act(obs)
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:  # display results every month
                current_month = info['month']
                print('Reward: ', sum(rewards), info)
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    env.close()

**************************************
Adding extra configuration definition
**************************************

You can even add a dictionary with extra parameters that modify the IDF you use before it is used in the simulations (or overwrite an existing one).

This new IDF version, which also adapts to the new weather you put in, is saved in the Sinergym output folder, leaving the original intact:

.. code:: python

    import gym
    import numpy as np

    import sinergym
    from sinergym.utils.rewards import LinearReward, ExpReward
    from sinergym.utils.wrapper import LoggerWrapper, NormalizeObservation
    from sinergym.utils.controllers import RBCDatacenter

    extra_conf={
        'timesteps_per_hour':6,
        'runperiod':(1,1,1991,2,1,1992),
        'action_definition': {
            'ThermostatSetpoint:DualSetpoint': [{
                'name': 'West-DualSetP-RL',
                'heating_name': 'West-HtgSetP-RL',
                'cooling_name': 'West-ClgSetP-RL',
                'zones': ['West Zone']
            },
                {
                'name': 'East-DualSetP-RL',
                'heating_name': 'East-HtgSetP-RL',
                'cooling_name': 'East-ClgSetP-RL',
                'zones': ['East Zone']
            }]
        }
    }

    env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1', 
                    reward=ExpReward, reward_kwargs={
                                        'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                        'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                        'range_comfort_winter': (20.0, 23.5),
                                        'range_comfort_summer': (23.0, 26.0),
                                        'energy_weight': 0.1},
                    weather_file='ESP_Granada.084190_SWEC.epw',
                    weather_variability=(1.0,0.0,0.001),
                    env_name='new_env_name',
                    act_repeat=4,
                    max_ep_data_store_num = 20,
                    config_params=extra_conf
                    )

    env = NormalizeObservation(env)
    env = LoggerWrapper(env)

    agent = RBCDatacenter(env)

    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
        while not done:
            a = agent.act(obs)
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:  # display results every month
                current_month = info['month']
                print('Reward: ', sum(rewards), info)
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    env.close()

.. note:: For more information on how each of the elements explained here works, please see the appropriate section.

.. note:: To see how Sinergym can be combined with DRL algorithms, please visit section :ref:`Deep Reinforcement Learning Integration` of our documentation (specifically the DRL_battery.py script in section :ref:`How use`).
