############################################
Extra Configuration in Sinergym simulations
############################################

With the `Modeling class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/config/modeling.py>`__, 
we have the ability to set up some **context details** in our simulation. This allows us to expand the 
context of each experiment and investigate more parameters. To use this functionality easily, you can 
provide these extra parameters in the **environment constructor** as follows:

.. code:: python

    import gymnasium as gym
    import sinergym

    extra_params={'timesteps_per_hour' : 6
                  'runperiod' : (1,1,1997,12,3,1998)}
    env = gym.make('Eplus-5Zone-hot-continuous-v1', config_params=extra_params)

The format for applying extra configuration is a **Python dictionary** with the extra parameter key name and its value.

.. note:: *Currently, only code skeleton and some parameters has been designed. Stay tuned for upcoming releases!*

Let's examine each implemented parameter for extra configuration separately:

******************
timestep_per_hour
******************

By default, *Sinergym* applies **4** timesteps per simulation hour, which is the default value in building files. 
However, you can modify this value using the **timestep_per_hour** key in the `config_params` dictionary and set 
more or fewer timesteps in each simulation hour.

******************
runperiod
******************

By default, a *Sinergym* simulation episode lasts one year (*from 1/1/1991 to 31/12/1991*). 
You can use the **runperiod** key to determine the **episode length** in the simulation. 
The format value for the **runperiod** key is a **tuple** with 
(*start_day*, *start_month*, *start_year*, *end_day*, *end_month*, *end_year*).

.. warning:: If you include a manual runperiod with this functionality, make sure not to include 
             February 29th of a leap year in that range. Otherwise, the simulator will fail, as 
             *EnergyPlus* does not account for leap days and the weather files do not include these days.

.. note:: More components may be managed in the future. Keep an eye out for upcoming releases!

.. note:: If you wish to create your own extra configuration parameters, refer to the method 
          ``apply_extra_conf`` in the `Modeling class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/config/modeling.py>`__.