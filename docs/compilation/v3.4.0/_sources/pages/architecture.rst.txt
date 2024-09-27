##############
Architecture
##############

As outlined in the introduction, *Sinergym* is structured as follows:

.. image:: /_static/sinergym_diagram.png
  :width: 800
  :alt: *Sinergym* backend
  :align: center

|

*Sinergym* consists of three primary components: the *agent*, the *communication* interface, and the *simulation*. 
The agent interacts with the environment by sending actions and receiving observations through the Gymnasium interface.
Concurrently, the Gymnasium interface communicates with the simulation engine via the *EnergyPlus* Python API. 
This API provides the functionality to manage handlers such as actuators, meters, and variables, whose current values 
directly influence the simulation's progression.

The following image provides a more detailed view of this process:

.. image:: /_static/backend.png
  :width: 1600
  :alt: *Sinergym* backend
  :align: center

|

Following the flow in figure, upon environment initialization (*__init__*),
a building model is created using the specified epJSON file (*1*), and a 
simulator instance is instantiated (*2*), receiving necessary data for its 
construction. Handlers for different components are initialized, which will 
be used in subsequent steps to gather information and control the simulation 
using the EnergyPlus Python API.

The next step is to initialize the first episode (*reset*), corresponding to the 
specified run period in the building. When starting a new episode, *Sinergym* uses 
its modeling class to adapt the specified building and weather to the environment (*3*) 
and start the simulation (*4*).

Key modeling actions include updating the EPW file if multiple are present, selecting 
one randomly per episode (*3.1*), automatically introducing Output Variables and Output 
Meters specified in the environment into the building model (*3.2*), adapting any extra 
configuration present in the environment, such as simulation timesteps per hour or custom 
run period (*3.3*), adjusting certain building aspects to be compatible with the weather 
specified in the environment, such as altitude, latitude, and building orientation (*3.4*), 
and finally applying weather variability if present (*3.5*).

Regarding the simulation, *Sinergym* stops the previous simulation if an episode was ongoing 
(*4.1*), generates the initial state (*4.2*), registers callbacks that interrupt the simulation 
to establish control flow (*4.3*), initiates the simulation process (*4.4*), and performs the 
warmup process (*4.5*).

Finally, the agent begins to send control actions to the building (*5*). *Sinergym* overwrites 
the selected schedulers incorporated in the building model, interrupting the simulation, 
setting the values, and resuming the process, thus adapting continuous control. Meanwhile, 
the middleware waits for the actions to be applied and the simulation state to change (*5.1*). 
Once this occurs, it is returned to the agent via the Gymnasium interface (*5.2*). 
This process is repeated continuously until the episode concludes (*step*).

This framework is highly abstract, as these components perform additional tasks such as managing the output's 
folder structure, preparing handlers for use, initiating callbacks for data collection during simulation, 
among other functions.

The architecture of *Sinergym* accomplishes the design objectives by offering many advantages. 
The framework is extensible since it facilitates the integration of custom reward functions, 
wrappers, controllers, and loggers, together with new EnergyPlus buildings and weathers. 
The use of EnergyPlus API as middleware allows access to all building sensors, metrics, 
actuators and other information during simulation. From the users' perspective, the framework 
hides the complexities of the interaction with the simulator to let them focus on data generation, 
learning control strategies and validation.


***********************************
Additional observation information
***********************************

Beyond the observations returned by the step and reset methods, as depicted in the preceding images, 
both methods also return a Python dictionary containing supplementary information:

- **Reset info:** This dictionary has the next keys:

.. code-block:: python

  info = {
            'time_elapsed(hours)': # <Simulation time elapsed in hours>,
            'month': # <Month in which the episode starts.>,
            'day': # <Day in which the episode starts.>,
            'hour': # <Hour in which the episode starts.>,
            'is_raining': # <True if it is raining in the simulation.>,
            'timestep': # <Timesteps count.>,
        }

- **step info:** This dictionary shares the same keys as the reset info, but additionally includes 
  the action dispatched (to the simulation, not the environment), the reward, and reward terms. The 
  reward terms are contingent on the reward function employed. For more details, refer to :ref:`Reward terms`.