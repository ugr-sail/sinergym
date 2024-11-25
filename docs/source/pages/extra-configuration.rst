###########################################
Extra configuration in Sinergym simulations
###########################################

Using the `Modeling class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/config/modeling.py>`__, it is possible to set up some **context details** of the simulation. This allows us to expand the context of each experiment with additional parameters. You can provide these extra configuration to the **environment constructor** as follows:

.. code:: python

    import gymnasium as gym
    import sinergym

    extra_params={'timesteps_per_hour' : 6
                  'runperiod' : (1,1,1997,12,3,1998)}
    env = gym.make('Eplus-5Zone-hot-continuous-v1', config_params=extra_params)

The format of these extra configuration parameters is a **Python dictionary** with their corresponding *keys* and *values*.

Let's examine each parameter separately.

*****************
timestep_per_hour
*****************

By default, *Sinergym* applies **4** timesteps per simulated hour, which is the default value in building files. 
However, you can modify this value using the ``timestep_per_hour`` key in the ``config_params`` dictionary and vary the number of timesteps in each simulated hour.

*********
runperiod
*********

By default, a *Sinergym* simulation episode lasts a single year (*from 1/1/1991 to 31/12/1991*). You can use the ``runperiod`` key to determine the **episode length** in the simulation. The format value for the ``runperiod`` key is a **tuple** with 
(``start_day``, ``start_month``, ``start_year``, ``end_day``, ``end_month``, ``end_year``).

.. warning:: If you include a manual ``runperiod``, make sure not to include 
             February 29th of a leap year in that range. Otherwise, the simulator will fail, as 
             *EnergyPlus* does not account for leap days and the weather files do not include these days.

.. note:: If you wish to create your own extra configuration parameters, refer to the method 
          ``apply_extra_conf`` in the `Modeling class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/config/modeling.py>`__.