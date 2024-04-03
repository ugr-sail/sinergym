###############
Output format
###############

When a simulation is running, it generates a directory named ``Eplus-env-<env_name>-res<num_simulation>``. 
The *Modeling* module also manages the directory tree generated during a simulation. The content of this 
root output directory is the result of the simulation and includes:

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

* **Eplus-env-sub_run<num_episode>** directories that record the results of each episode in the simulation. 
  The number of these directories depends on the number of episodes and the *maximum episode data value* 
  (see :ref:`Maximum Episode Data Stored in Sinergym Output`).

* Within these directories, the structure is always the same:

    * A copy of the **environment.epJSON** used during the simulation episode. This does not have to be the same 
      as the original hosted in the repository, as the simulation can be modified to suit specific user-defined 
      settings when building the gymnasium environment.

    * A copy of the **Weather.epw** used during the simulation episode. This file does not have to be the same as 
      the original (when using variability).

    * **monitor.csv**: This records all Agent-Environment interactions during the episode, timestep by timestep. 
      This file only exists when the environment has been wrapped with **Logger** (see :ref:`Wrappers` for more information).

    * **monitor_normalized.csv**: This file is only generated when the environment is wrapped 
      with **logger and normalization** (see :ref:`Wrappers`). The structure is the same as **monitor.csv**, 
      but the ``observation_values`` are normalized.

    * **output/**: This directory contains the **EnergyPlus simulation output**. To learn more about these files, 
      visit the `EnergyPlus documentation <https://energyplus.net/documentation>`__.

* **data_available.txt**: This file is generated when the *EnergyPlus* API initializes all callbacks and handlers 
  for the simulation. In this file, you can find all the available components of the building model, such as 
  actuators, schedulers, meters, variables, internal variables, etc.

  .. warning:: Some lists of components, such as ``Output:Variable``'s, do not fully appear in *data_available.txt* 
               because they must be declared in the building model first. If you want to see all the variables 
               or meters specifically, you should look for them in the correct *Energyplus* output file. 
               If you specify a correct variable in the environment, *Sinergym* will add the ``Output:Variable`` 
               element in the building model before the simulation starts.
                   
* **progress.csv**: This file contains information about general simulation results. There is a **row per episode** 
  and it records important data such as mean power consumption or mean comfort penalty, for example. This file 
  only exists when the environment has been wrapped with **Logger** (see :ref:`Wrappers` for more information).

****************
Logger
****************

The files **monitor.csv**, **monitor_normalized.csv**, and **progress.csv** belong to the **Sinergym logger**, 
which is a wrapper for the environment. This logger is responsible for recording all the interactions that 
occur in a simulation, regardless of the training technique being used or any other external factor.

Recording is managed by an instance of the ``CSVLogger`` class, which is present as a wrapper attribute and 
is called at each timestep and at the end of an episode. This class can be replaced by a new one 
(see :ref:`Logger Wrapper personalization/configuration`).

.. warning:: The ``CSVLogger`` requires the info dict with specific keys to log the information correctly. 
             If you change the info dict structure in Sinergym, you should check this logger or use a custom one.

.. note:: Normalized observation methods are only used when the environment is wrapped with normalization previously.

.. note:: You can activate and deactivate the logger from the environment whenever you want, using the activate 
          and deactivate methods, so you don't need to unwrap the environment.