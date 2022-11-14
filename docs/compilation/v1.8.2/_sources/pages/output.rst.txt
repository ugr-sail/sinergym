###############
Output format
###############

When a simulation is running, this generates a directory called ``Eplus-env-<env_name>-res<num_simulation>``. The content of this directory is the result of the simulation and we have:

::

    Eplus-env-<env_name>-res<num_simulation>
    ├── Eplus-env-sub_run1
    ├── Eplus-env-sub_run2
    ├── Eplus-env-sub_run3
    ├── ...
    ├── Eplus-env-sub_runN
    │   ├── output/
    │   ├── variables.cfg
    │   ├── socket.cfg
    │   ├── utilSocket.cfg
    │   ├── environment.idf
    |   ├── weather.epw
    │   ├── monitor.csv
    |   └── monitor_normalized.csv (optional)
    └── progress.csv

* **Eplus-env-sub_run<num_episode>** records the results of each episode in simulation. The number of these directories depends on the number of episodes.
* Within these directories, you have always the same structure:
    * A copy of **variables.cfg** and **environment.idf** which are being used during simulation. **Environment.idf** does not have to be the same as the original hosted in the repository. Since the simulation can be modified to suit the specific weather or apply extra user-defined settings when building the gym environment.
    * A copy of **Weather.epw** appears only when the weather change for one episode to another (using variability, for example). If weather does not change, original repository *.epw* will be used in each episode.
    * A copy of **socket.cfg** and **utilSocket.idf** which are being used in order to  establish communication interface with Energyplus during simulation.
    * **monitor.csv**: This records all interactions Agent-Environment during the episode timestep by timestep, the format is: *timestep, observation_values, action_values, simulation_time (seconds), reward, done*. This file only exists when environment has been wrapped with **Logger** (see :ref:`Wrappers` for more information).
    * **monitor_normalized.csv**: This file is only generated when environment is wrapped with **logger and normalization** (see :ref:`Wrappers`). The structure is the same than **monitor.csv** but ``observation_values`` are normalized.
    * **output/**: This directory has **EnergyPlus environment output**.
* **progress.csv**: This file has information about general simulation results. There is a row per episode and it records most important data. Currently, the format is: *episode_num,cumulative_reward,mean_reward,cumulative_power_consumption,
  mean_power_consumption,cumulative_comfort_penalty,mean_comfort_penalty,
  cumulative_power_penalty,mean_power_penalty,comfort_violation (%),length(timesteps),
  time_elapsed(seconds)*. This file only exists when environment has been wrapped with **Logger** (see :ref:`Wrappers` for more information).

.. note:: For more information about specific EnergyPlus output, visit `EnergyPlus documentation <https://energyplus.net/documentation>`__.

****************
Logger
****************

The files **monitor.csv**, **monitor_normalized.csv** and **progress.csv** belong to **Sinergym logger** which is a wrapper for the environment (see :ref:`Wrappers`). This logger has the responsibility of recording 
all the interactions that are carried out in a simulation,
regardless of the training technique which may be being used or any other external factor.

Recording is managed by a instance of the class ``CSVLogger`` which is present as a environment attribute and is called in each timestep and in the end of a episode:

.. literalinclude:: ../../../sinergym/utils/common.py
    :language: python
    :pyobject: CSVLogger

.. note:: Normalized observation methods are only used when environment is wrapped with normalization previously (see :ref:`Wrappers`).

.. note:: Note that you can activate and deactivate logger from environment when you want it, using methods activate and deactivate, so you don't need to unwrap environment.


