The goal of *Sinergym* is to create an environment following *Gymnasium* interface for wrapping simulation engines (*EnergyPlus*) for building control using
**deep reinforcement learning**.

.. image:: /_static/general_blueprint.png
  :width: 800
  :alt: *Sinergym* diagram
  :align: center

|

.. raw:: html
    :file: ../_templates/shields.html

.. note:: Please, help us to improve by **reporting your questions and issues** 
   `here <https://github.com/ugr-sail/sinergym/issues>`__. It is easy, just 2 clicks 
   using our issue templates (questions, bugs, improvements, etc.). More detailed 
   info on how to report issues 
   `here <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`__. 

The main functionalities of *Sinergym* are the following:

-  **Compatibility with simulation engines**. Communication between
   Python and `EnergyPlus <https://energyplus.net/>`__ is established
   using `Energyplus Python API <https://energyplus.readthedocs.io/en/latest/api.html>`__ as middleware.
   However, more of them (e.g.
   `OpenModelica <https://openmodelica.org/>`__) could be included in
   the backend while maintaining the Gymnasium API in the future.

-  **Benchmark environments**. Similarly to *Atari* or *Mujoco* environments
   for RL community, we are designing a set of environments for
   benchmarking and testing deep RL algorithms. These environments may
   include different buildings, weathers, action/observation spaces, function rewards, etc.

-  **Customizable environments**. We aim to provide a
   package which allows modifying experimental settings in an easy
   manner. The user can create his own environments, combining his own
   building model, weather, reward, observation/action space, variables, actuators, environment name, etc.
   The user can also use these pre-configured environments available in *Sinergym* 
   and change some aspect of it (for example, the weather) in such 
   a way that he does not  have to make an entire definition of the 
   environment and can start from one pre-designed by us.
   Some parameters directly associated with the simulator can be set as **extra configuration** 
   as well, such as people occupant, time-steps per simulation hour, run-period, etc.

-  **Customizable components**. *Sinergym* is easily scalable by third parties.
   Following the structure of the implemented classes, new custom components 
   can be created for new environments such as function rewards, wrappers,
   controllers, etc.

-  **Automatic Building Model adaptation to user changes**. Many of the updates to the environment definition require changes 
   to the building model (*epJSON* file) to adapt it to these new features before the simulation starts, which *Sinergym* will 
   perform automatically. For example, using another weather file requires building location and design days update, using new 
   observation variables requires to update the ``Output:Variable`` and ``Output:Meter`` fields, the same occurs with extra 
   configuration context concerned with simulation directly, if weather variability is set, then a weather with noise 
   will be used. These new building and weather file versions, is saved in the *Sinergym* output folder, leaving the original 
   intact. In short, *Sinergym* automates the whole process of model adaptation so that the user 
   only has to define what he wants for his environment.

-  **Automatic actuators control**. Related to the above, it will only be necessary to specify the name of the actuators to be controlled 
   through the actions of the Gymnasium interface, and *Sinergym* will take care of everything.

-  **Extensive environment information**. It is important that users can get some information about *Sinergym* background components from environment interface easily.
   From environment instance, it is possible to consult available schedulers, variables which compose an observation and action, whether simulator is running,
   the building run period, episode length, timesteps per episode, available building zones... And much more.

-  **Stable Baseline 3 Integration**. Some functionalities like callbacks
   have been customized by our team in order to test easily these environments
   with deep reinforcement learning algorithms and logger specific information about 
   *Sinergym* environments. 
   However, *Sinergym* is completely agnostic to any DRL algorithm and can be used with any DRL 
   library that works with gymnasium interface.

-  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with *Sinergym*, we tell you some details about how to do it.

-  **Weights & Biases tracking and visualization compatibility**. One of *Sinergym*'s objectives is to automate
   and facilitate the training, reproducibility and comparison of agents in simulation-based 
   building control problems, managing and monitoring model lifecycle from training to deployment. `WandB <https://wandb.ai/site>`__
   is an open-source platform for the machine learning lifecycle helping us with this issue. 
   It lets us register experiments hyperparameters, visualize data recorded in real-time, 
   and store artifacts with experiment outputs and best obtained models. 

-  **Notebooks examples**. *Sinergym* develops code in notebook format with the purpose of offering use cases to 
   the users in order to help them become familiar with the tool. They are constantly updated, along with the updates 
   and improvements of the tool itself.

-  This project is accompanied by extensive **documentation**, **unit tests** and **github actions workflows** to make 
   *Sinergym* an efficient ecosystem for both understanding and development.

-  Many more!

.. important:: If you want to introduce your own buildings in *Sinergym*, please visit :ref:`Adding new buildings for environments` section.

.. note:: *This is a work in progress project. Stay tuned for upcoming releases!*
