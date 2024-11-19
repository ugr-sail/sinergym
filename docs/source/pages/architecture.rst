############
Architecture
############

*Sinergym* is structured as follows:

.. image:: /_static/sinergym_diagram.png
  :width: 800
  :alt: *Sinergym* backend
  :align: center

|

*Sinergym* is composed of three main components: the *agent*, the *communication interface*, and the *simulation*.  
The agent interacts with the environment by sending actions and receiving observations through the Gymnasium interface.  
Simultaneously, the Gymnasium interface establishes communication with the simulation engine using the *EnergyPlus* Python API.  
This API enables the management of handlers such as variables, meters, and actuators, whose values directly impact the simulation's execution.

The following image provides a detailed view of this process:

.. image:: /_static/backend.png
  :width: 1600
  :alt: *Sinergym* backend
  :align: center

|

### Environment initialization

During environment initialization (*__init__*), a building model is created using the specified epJSON file (*1*), and a simulator instance is instantiated (*2*), receiving the necessary data for its creation. Handlers for various components are initialized to facilitate the collection of information and control of the 
simulation via the EnergyPlus Python API.

The next step involves initializing the first episode (*reset*), corresponding to the specified run period for the building. When a new episode begins, *Sinergym* utilizes its `modeling <https://github.com/ugr-sail/sinergym/blob/main/sinergym/config/modeling.py>`__ class to adapt the designated building and weather to the environment (*3*) and start the simulation (*4*). Key steps include:

- **Updating the EPW file** if multiple files are used, randomly selecting one per episode (*3.1*).  
- **Introducing Output Variables and Output Meters** specified in the environment into the building model (*3.2*).  
- **Adapting extra configurations**, such as simulation timesteps per hour or the custom run period (*3.3*).  
- **Adjusting building attributes** (e.g., altitude, latitude, and orientation) to align with the specified environment weather (*3.4*).  
- **Applying weather variability**, if defined (*3.5*). This process involves adding random noise to the weather dataset to simulate varying weather conditions during each run period.

### Simulation startup  

During simulation startup, *Sinergym* executes the following steps:  

- Stops any ongoing simulation (*4.1*).  
- Generates the initial state (*4.2*).  
- Registers callbacks to interrupt the simulation for control flow (*4.3*).  
- Initiates the simulation process (*4.4*).  
- Executes the warmup process (*4.5*).  

### Agent-environment interaction

Once the simulation is ready, the agent begins sending control actions to the building (*5*). *Sinergym* replaces the building model's predefined schedulers by interrupting the simulation, applying the agent's actions, and resuming the process, thus enabling continuous control. The middleware coordinates this by waiting for the actions to take effect and the simulation state to update (*5.1*). Once updated, the new state is returned to the agent via the Gymnasium interface (*5.2*) and interaction continues via *step*. This cycle repeats until the episode concludes.

|

The framework is designed to handle additional tasks seamlessly, such as managing output directory structures, preparing handlers, and collecting data during the simulation using callbacks.  

The *Sinergym* architecture achieves its design goals by providing several advantages:  

- **Extensibility**. Support to custom reward functions, wrappers, controllers, and loggers, as well as new EnergyPlus building and weather configurations.  
- **Comprehensive middleware**. The EnergyPlus API grants access to all building sensors, metrics, actuators, and related simulation data.  
- **User-Friendly abstraction**. The framework abstracts away the complexities of simulator interaction, allowing users to focus on data generation, learning control strategies, and validation.

***********************************
Additional observation information
***********************************

In addition to the observations returned by the *step* and *reset* methods, as illustrated in the preceding images, both methods also provide a Python dictionary containing supplementary information:

- *reset* info. This dictionary contains the following keys:

.. code-block:: python

  info = {
            'time_elapsed(hours)': # <Simulation time elapsed (in hours)>,
            'month': # <Month in which the episode starts.>,
            'day': # <Day in which the episode starts.>,
            'hour': # <Hour in which the episode starts.>,
            'is_raining': # <True if it is raining in the simulation.>,
            'timestep': # <Timesteps count.>,
        }

- *step* info. This dictionary shares the same keys as the reset info, but aslo includes 
  the action dispatched (to the simulation, not the environment), the reward, and the reward terms, which depend on the reward function employed. For more details, refer to :ref:`Reward terms`.