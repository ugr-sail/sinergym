############################################
Extra Configuration in Sinergym simulations
############################################

Using `Config class <https://github.com/jajimer/sinergym/tree/main/sinergym/utils/config.py>`__ in simulator, we have the possibility to set up some details in our simulation. This let us to amplify the context of each experiment and have more parameters to investigate.
To use this functionality easily, you can provide this extra parameters in env constructor in this way:

.. code:: python

    import gym
    import sinergym

    extra_params={'timesteps_per_hour' : 6}
    env = gym.make('Eplus-5Zone-hot-continuous-v1', config_params=extra_params)

In this example, by default a Sinergym simulation apply 4 timestep per simulation hour. Sinergym will modify this simulation model from Python code and save IDF in each episode directory generated in output. For more information, see :ref:`Output format`.
The format for apply extra configuration is a Python dictionary with extra parameter key name and value.

.. note:: *Currently, only code skeleton and timesteps_per_hour parameter has been designed. Stay tuned for upcoming releases!*