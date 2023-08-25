###############
Output format
###############

When a simulation is running, this generates a directory called 
``Eplus-env-<env_name>-res<num_simulation>``. The management of 
the directories tree generated during a simulation is done by 
the *Modeling* module too. The content of this root output directory 
is the result of the simulation and we have:

::

    Eplus-env-<env_name>-res<num_simulation>
    ├── Eplus-env-sub_run1
    ├── Eplus-env-sub_run2
    ├── Eplus-env-sub_run3
    ├── ...
    ├── Eplus-env-sub_runN
    │   ├── output/
    │   ├── environment.epJSON
    |   ├── weather.epw
    │   ├── monitor.csv
    |   └── monitor_normalized.csv (optional)
    ├── data_available.txt
    └── progress.csv

* **Eplus-env-sub_run<num_episode>** records the results of each episode in 
  simulation. The number of these directories depends on the number of episodes
  and *maximum episode data value* 
  (see :ref:`Maximum Episode Data Stored in Sinergym Output`).

* Within these directories, you have always the same structure:

    * A copy of **environment.epJSON** which is being used during 
      simulation episode. **Environment.epJSON** does not have to be the same as the original 
      hosted in the repository. Since the simulation can be modified to suit the 
      specific user-defined settings when building the gymnasium environment.

    * A copy of **Weather.epw** which is being used during 
      simulation episode. This file does not have to be the 
      same than original (when using variability).

    * **monitor.csv**: This records all interactions Agent-Environment during 
      the episode timestep by timestep. This file only exists 
      when environment has been wrapped with **Logger** (see :ref:`Wrappers` for 
      more information).

    * **monitor_normalized.csv**: This file is only generated when environment is 
      wrapped with **logger and normalization** (see :ref:`Wrappers`). The structure 
      is the same than **monitor.csv** but ``observation_values`` are normalized.

    * **output/**: This directory has **EnergyPlus simulation output**.
      If you want to know more about this files, visit 
      `EnergyPlus documentation <https://energyplus.net/documentation>`__.

* **data_available.txt**: This file is generated when *EnergyPlus* API initializes all
  callbacks and handlers for the simulation. In this file, we can find all the available
  components of the building model such as actuators, schedulers, meters, variables, internal
  variables, etc.

  .. warning:: Some list of components such as ``Output:Variable``'s does not appear fully in
               *data_available.txt*, because of it must be declared in the building model first.
               If you want to see all the variables or meters specifically, you should look for them
               in the correct *Energyplus* output file. If you specify a correct variable in environment,
               *Sinergym* will add the ``Output:Variable`` element in the building model before simulation start.
                   
* **progress.csv**: This file has information about general simulation results. 
  There is a **row per episode** and it records most important data such as mean 
  power consumption or mean comfort penalty, for example. This file only 
  exists when environment has been wrapped with 
  **Logger** (see :ref:`Wrappers` for more information).

****************
Logger
****************

The files **monitor.csv**, **monitor_normalized.csv** and **progress.csv** 
belong to **Sinergym logger** which is a wrapper for the environment. 
This logger has the responsibility of recording 
all the interactions that are carried out in a simulation,
regardless of the training technique which may be being used or any other 
external factor.

Recording is managed by an instance of the class ``CSVLogger`` which is 
present as a wrapper attribute and is called in each timestep and 
in the end of a episode. This class can be substitute by a new one,
see :ref:`Logger Wrapper personalization/configuration`.

.. note:: Normalized observation methods are only used when environment is 
          wrapped with normalization previously.

.. note:: Note that you can activate and deactivate logger from environment 
          when you want it, using methods activate and deactivate, so 
          you don't need to unwrap environment.