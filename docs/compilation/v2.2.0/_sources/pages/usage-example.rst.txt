#############
Usage example
#############

If you used our Dockerfile during installation, you should have the *try_env.py* file in your workspace as soon as you enter in. In case you have installed everything on your local machine directly, place it inside our cloned repository.
In any case, we start from the point that you have at your disposal a terminal with the appropriate python version and Sinergym running correctly.

At this point of the documentation, we have explained how to install Sinergym, the environments it includes, the reward functions we can define, wrappers and controllers.

In this section we will see some examples of use to better understand how they can be used in practice.

If you want to execute our notebooks on your own it is possible in `examples` folder.

*****************
Simplest example
*****************

Let's start with the simplest use case for the Sinergym tool. In the root repository we have the script **try_env.py**:

.. literalinclude:: ../../../scripts/try_env.py
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

.. warning:: When specifying a different reward with `gym.make` than the default environment ID, it is very important to set the `reward_kwargs` that are required and therefore do not have a default value. In the rewards we have defined it is required: **temperature_variable(s)**, **energy_variable**, **range_comfort_winter**, **range_comfort_summer**. The rest of them have default values and it is not necessary to specify.

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
    from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation

    env = gym.make('Eplus-5Zone-hot-continuous-v1')
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
    from sinergym.utils.controllers import RBC5Zone

    env = gym.make('Eplus-5Zone-hot-continuous-v1')

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
Overwriting some default values of the environments
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
    from sinergym.utils.controllers import RBCDatacenter

    env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1', 
                    weather_file='ESP_Granada.084190_SWEC.epw',
                    weather_variability=(1.0,0.0,0.001),
                    env_name='new_env_name',
                    act_repeat=4,
                    max_ep_data_store_num = 20)

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

*******************************************
Overwriting observation and action spaces
*******************************************

By default, the IDs of the predefined environments in *Sinergym* already have a space of actions and observations set.

However, it can be overwritten by a new definition of them. On the one hand, we will have to define the name of the 
**variables**, and on the other hand, the definition of the **spaces** (and an **action mapping** if it is a discrete environment).

.. code:: python

    import gym
    import numpy as np

    import sinergym

    new_observation_variables=[
        'Site Outdoor Air Drybulb Temperature(Environment)',
        'Site Outdoor Air Relative Humidity(Environment)',
        'Site Wind Speed(Environment)',
        'Zone Thermal Comfort Fanger Model PPD(East Zone PEOPLE)',
        'Zone People Occupant Count(East Zone)',
        'People Air Temperature(East Zone PEOPLE)',
        'Facility Total HVAC Electricity Demand Rate(Whole Building)'
    ]

    new_action_variables = [
        'West-HtgSetP-RL',
        'West-ClgSetP-RL',
        'East-HtgSetP-RL',
        'East-ClgSetP-RL'
    ]

    new_observation_space = gym.spaces.Box(
        low=-5e6,
        high=5e6,
        shape=(len(new_observation_variables) + 4,),
        dtype=np.float32)

    new_action_mapping = {
        0: (15, 30, 15, 30),
        1: (16, 29, 16, 29),
        2: (17, 28, 17, 28),
        3: (18, 27, 18, 27),
        4: (19, 26, 19, 26),
        5: (20, 25, 20, 25),
        6: (21, 24, 21, 24),
        7: (22, 23, 22, 23),
        8: (22, 22, 22, 22),
        9: (21, 21, 21, 21)
    }

    new_action_space = gym.spaces.Discrete(10)

    env = gym.make('Eplus-datacenter-cool-discrete-stochastic-v1', 
                   observation_variables=new_observation_variables,
                   observation_space=new_observation_space,
                   action_variables=new_action_variables,
                   action_mapping=new_action_mapping,
                   action_space=new_action_space
                )


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

In case the definition has some inconsistency, such as the *IDF* has not been 
adapted to the new actions, the spaces do not fit with the variables, the 
observation variables do not exist, etc. Sinergym will display an error.

********************************
Adding a new action definition
********************************

As we have explained in the previous example, one of the problems that can arise when 
modifying the space of actions and observations is that the *IDF* is not adapted to the 
new space of actions established.

We may even want to modify the effects of actions on the building directly for some kind 
of interest without being subject to a change of the action space. For example, we may 
want to change the zones assigned to each thermostat or change their value at the start 
of the simulation.

For this purpose, the *Sinergym* **action definition** is available. With a dictionary we can 
build a definition of what we want to be controlled in the building and how to control 
it using the action space of the environment:

.. code:: python

    import gym
    import numpy as np

    import sinergym

    new_action_definition={
        'ThermostatSetpoint:DualSetpoint': [{
            'name': 'West-DualSetP-RL',
            'heating_name': 'West-HtgSetP-RL',
            'cooling_name': 'West-ClgSetP-RL',
            'heating_initial_value':21.0,
            'cooling_initial_value':25.0,
            'zones': ['West Zone']
        },
            {
            'name': 'East-DualSetP-RL',
            'heating_name': 'East-HtgSetP-RL',
            'cooling_name': 'East-ClgSetP-RL',
            'heating_initial_value':21.0,
            'cooling_initial_value':25.0,
            'zones': ['East Zone']
        }]
    }

    env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1', 
                    action_definition=new_action_definition
                    )

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

The name of the heating and cooling should be the name of action variables defined in the environment. 
Otherwise, Sinergym will show the inconsistency.

For more information about the format of the action definition dictionaries, visit the section :ref:`Action definition`.

**************************************
Adding extra configuration definition
**************************************

You can even add a dictionary with extra parameters that modify the IDF you use before it is 
used in the simulations (or overwrite an existing one).

This new IDF version, which also adapts to the new weather you put in, is saved in the *Sinergym* 
output folder, leaving the original intact:

.. code:: python

    import gym
    import numpy as np

    import sinergym

    extra_conf={
        'timesteps_per_hour':6,
        'runperiod':(1,1,1991,2,1,1992)
    }

    env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1', 
                    config_params=extra_conf
                    )

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

.. note:: For more information on how each of the elements explained here works, please see the appropriate section.

.. note:: To see how Sinergym can be combined with DRL algorithms, please visit section :ref:`Deep Reinforcement Learning Integration` of our documentation (specifically the DRL_battery.py script in section :ref:`How use`).

.. note:: Our team provide several notebooks with more functionality and examples, visit examples section.
