###############
Output format
###############

When a simulation is run, this generate a directory called `Eplus-env-<env_name>-res<num_simulation>`. The content of this directory is the result of the simulation and we have:

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
    │   ├── enviroment.idf
    │   └── monitor.csv
    └── progress.csv

- **Eplus-env-sub_run<num_episode>** records the results of each episode in simulation. The number of these directories depends on the number of episodes.
- Within these directories, you have always the same structure:
	- A copy of **variables.cfg** and **environment.idf** which are being used during simulation.
	- A copy of **socket.cfg** and **utilSocket.idf** which are being used in order to communication interface with Energyplus during simulation.
	- **monitor.csv**: This records all interactions Agent-Enviroment during the episode timestep by timestep, the format is: *timestep, observation_values, action_values, simulation_time (seconds), reward, done*.
	- **output/**: This directory has EnergyPlus environment output.
- **progress.csv**: This file has information about general simulation results. There is a row per episode and it records most important data. Currently, the format is: *episode,ep_mean_reward,cumulative_reward,total_time_elapsed*.

.. note:: For more information about EnergyPlus output, visit `EnegyPlus documentation <https://energyplus.net/documentation>`__.
