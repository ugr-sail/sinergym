############################################
Extra Configuration in Sinergym simulations
############################################

Using `Config class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/config.py>`__ in simulator, we have the possibility to set up some details in our simulation. This let us to amplify the context of each experiment and have more parameters to investigate.
To use this functionality easily, you can provide this extra parameters in env constructor in this way:

.. code:: python

    import gym
    import sinergym

    extra_params={'timesteps_per_hour' : 6
                  'runperiod' : (1,1,1997,12,3,1998)}
    env = gym.make('Eplus-5Zone-hot-continuous-v1', config_params=extra_params)

Sinergym will modify this simulation model from Python code and save IDF in each episode directory generated in output. For more information, see :ref:`Output format`.
The format for apply extra configuration is a **Python dictionary** with extra parameter key name and value.

.. note:: *Currently, only code skeleton and some parameters has been designed. Stay tuned for upcoming releases!*

Let's see each implemented parameter for extra configuration separately:

******************
timestep_per_hour
******************

By default, a Sinergym simulation apply 4 timestep per simulation hour. However, you have the possibility to modify this value using **timestep_per_hour** key in `config_params` dictionary and set more/less timesteps in each simulation hour.

******************
runperiod
******************

By default, a Sinergym simulation episode is one year (from 1/1/1991 to 31/12/1991). You can use this **runperiod** key and, as a result, determine episode length in simulation. 
The format value for **runperiod** key is a **tuple** with (*start_day*, *start_month*, *start_year*, *end_day*, *end_month*, *end_year*).

.. warning:: If we include a manual runperiod with this functionality, we should not include any February 29th of a leap year in that range. Otherwise, the simulator will fail, since Energyplus does not take into account leap days and the weather files do not include these days.

******************
action_definition
******************

Creating a **new external interface** to control different parts of a building is not a trivial task, it requires certain changes in the building model (IDF), 
configuration files for the external interface (variables.cfg), etc in order to control it.

For this purpose,  we have available **action_definition** key extra parameter. Its value is a dictionary with the next structure:

.. code:: python

    extra_params={
        'action_definition':{
            <controller_type>:[<controller1_definition>,<controller2_definition>, ...]
        }
    }

The `<controller_definition>` will depend on the specific type of controller that we are going to create, we have the next support:

Thermostat:DualSetpoint
~~~~~~~~~~~~~~~~~~~~~~~~

This controller has the next values in its definition:

- *name*: DualSetpoint resource name (str).
- *heating_name*: Heating setpoint name. This name should be an action variable defined in your environment (str).
- *cooling_name*: Cooling setpoint name. This name should be an action variable defined in your environment (str).
- *zones*: An thermostat can manage several building zones at the same time. Then, you can specify one or more zones (List(str)). If the zone name specified is not exist in building, Sinergym will report the error.

For an example about how to use it, see :ref:`Adding extra configuration definition`.

.. note:: Actually, we only support `Thermostat:DualSetpoint` definition, but more components could be managed in the future. Stay tuned for upcoming releases! 

.. note:: If you want to create your own extra configuration parameters, please see the method `apply_extra_conf` from `Config class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/config.py>`__.