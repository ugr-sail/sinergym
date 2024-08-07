###############
Output format
###############

When a simulation is running, it generates a directory named ``Eplus-env-<env_name>-res<num_simulation>``. 
The *Modeling* module also manages the directory tree generated during a simulation. The content of this 
root output directory is the result of the simulation and includes:

.. image:: /_static/output_structure.png
  :scale: 50 %
  :alt: Logger wrappers graph.
  :align: center

* **Eplus-env-sub_run<num_episode>** directories that record the results of each episode in the simulation. 
  The number of these directories depends on the number of episodes and the *maximum episode data value* 
  (see :ref:`Maximum Episode Data Stored in Sinergym Output`). Within these directories, the structure is 
  always the same:

    * A copy of the **environment.epJSON** used during the simulation episode. This does not have to be the same 
      as the original hosted in the repository, as the simulation can be modified to suit specific user-defined 
      settings when building the gymnasium environment.

    * A copy of the **Weather.epw** used during the simulation episode. This file does not have to be the same as 
      the original (when using variability).

    * **monitor/**: This records all Agent-Environment interactions during the episode, timestep by timestep.
      This directory only exists when the environment has been wrapped with **LoggerWrapper** and **CSVLgger** (see :ref:`Wrappers` for more information).
      This directory as several CSV files depending on the data:
        
        * **observations.csv**: This file contains the observations of the environment at each timestep. The header 
          is the observation variables names. 
        
        * **agent_actions.csv**: This file contains the actions taken by the agent at each timestep. The header is the
          action variables names.

        * **simulated_actions.csv**: This file contains the actions executed in the simulation at each timestep. This values
          do not have to be the same as the agent actions, for example, when the environment has been wrapped with normlization 
          in its action space.The header is the action variables names.

        * **rewards.csv**: This file contains the rewards obtained by the agent at each timestep.

        * **infos.csv**: This file contains the info dict of the environment at each timestep. The header is the info dict keys.
          Some of the info keys can be ignored depending on the **CSVLogger** configuration.

        * **normalized_observations.csv**: This file is only generated when the environment is wrapped with **NormalizeObservation** 
          (see :ref:`Wrappers`). The structure is the same as **observations.csv**, but the values are normalized.

        * **custom_metrics.csv**: This file only appears when the logger wrapper has a definition of this custom metrics. The header
          is the custom metric list names specified.

    * **output/**: This directory contains the **EnergyPlus simulation output**. To learn more about these files, 
      visit the `EnergyPlus Output File List <https://bigladdersoftware.com/epx/docs/24-1/output-details-and-examples/output-file-list.html#output-file-list>`__.

* **progress.csv**: This file contains information about general simulation results. There is a **row per episode** 
  and it records important data such as mean power consumption, mean reward or mean comfort penalty, for example. This file 
  only exists when the environment has been wrapped with a **LoggerWrapper** and **CSVLogger**, same as monitor folder 
  (see :ref:`Wrappers` for more information). The exact structure of this file is defined by the **LoggerWrapper** class.

* **data_available.txt**: This file is generated when the *EnergyPlus* API initializes all callbacks and handlers 
  for the simulation. In this file, you can find all the available components of the building model, such as 
  actuators, schedulers, meters, variables, internal variables, etc.

  .. warning:: Some lists of components, such as ``Output:Variable``'s, do not fully appear in *data_available.txt* 
               because they must be declared in the building model first. If you want to see all the variables 
               or meters specifically, you should look for them in the correct *Energyplus* output file. 
               If you specify a correct variable in the environment, *Sinergym* will add the ``Output:Variable`` 
               element in the building model before the simulation starts.