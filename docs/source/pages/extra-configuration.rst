############################################
Extra Configuration in Sinergym simulations
############################################

Using `Config class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/config.py>`__ 
in simulator, we have the possibility to set up some **details of context** in our simulation. 
This let us to amplify the context of each experiment and have more parameters to investigate.
To use this functionality easily, you can provide this extra parameters in **environment constructor** in this way:

.. code:: python

    import gymnasium as gym
    import sinergym

    extra_params={'timesteps_per_hour' : 6
                  'runperiod' : (1,1,1997,12,3,1998)}
    env = gym.make('Eplus-5Zone-hot-continuous-v1', config_params=extra_params)

*Sinergym* will modify this simulation model from Python code and save *epJSON* in each 
episode directory generated in output. For more information, see :ref:`Output format`.
The format for apply extra configuration is a **Python dictionary** with extra parameter key name and value.

.. note:: *Currently, only code skeleton and some parameters has been designed. Stay tuned for upcoming releases!*

Let's see each implemented parameter for extra configuration separately:

******************
timestep_per_hour
******************

By default, a *Sinergym* simulation apply **4** timestep per simulation hour. However, 
you have the possibility to modify this value using **timestep_per_hour** key 
in `config_params` dictionary and set more/less timesteps in each simulation hour.

******************
runperiod
******************

By default, a *Sinergym* simulation episode is one year (*from 1/1/1991 to 31/12/1991*). 
You can use this **runperiod** key and, as a result, determine **episode length** in simulation. 
The format value for **runperiod** key is a **tuple** with 
(*start_day*, *start_month*, *start_year*, *end_day*, *end_month*, *end_year*).

.. warning:: If we include a manual runperiod with this functionality, we should not include any 
             February 29th of a leap year in that range. Otherwise, the simulator will fail, 
             since *EnergyPlus* does not take into account leap days and the weather files 
             do not include these days.

.. note:: More components could be managed in the future. Stay tuned for upcoming releases! 

.. note:: If you want to create your own extra configuration parameters, 
          please see the method ``apply_extra_conf`` from 
          `Config class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/config.py>`__.