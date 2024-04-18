The goal of *Sinergym* is to create an environment following *Gymnasium* interface for wrapping 
simulation engines (*EnergyPlus*) for building control using **deep reinforcement learning** 
or any external control.


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

*Sinergym* offers:

-  **Simulation Engine Compatibility**: Uses `EnergyPlus Python API <https://energyplus.readthedocs.io/en/latest/api.html>`__ 
   for Python-EnergyPlus communication. Future plans include more engines like `OpenModelica <https://openmodelica.org/>`.

-  **Benchmark Environments**: Designs environments for benchmarking and testing deep RL algorithms or other external strategies, 
   similar to *Atari* or *Mujoco*.

-  **Customizable Environments**: Allows easy modification of experimental settings. Users can create 
   their own environments or modify pre-configured ones in *Sinergym*.

-  **Customizable Components**: Enables creation of new custom components for new environments, 
   making *Sinergym* scalable, such as function rewards, wrappers,controllers, etc. 

-  **Automatic Building Model Adaptation**: *Sinergym* automates the process of adapting the 
   building model to user changes in the environment definition.

-  **Automatic Actuators Control**: Controls actuators through the Gymnasium interface 
   based on user specification, only actuators names are required and *Sinergym* will
   do the rest.

-  **Extensive Environment Information**: Provides comprehensive information about *Sinergym* background components 
   from the environment interface.

-  **Stable Baseline 3 Integration**: Customizes functionalities for easy testing of environments with SB3 algorithms, 
   such as callbacks and cutomizable training real-time logging. However, *Sinergym* is agnostic to any DRL algorithm.

-  **Google Cloud Integration**: Offers guidance on using *Sinergym* with Google Cloud infrastructure.

-  **Weights & Biases Compatibility**: Automates and facilitates training, reproducibility, and 
   comparison of agents in simulation-based building control problems. `WandB <https://wandb.ai/site>`__ 
   assists in managing and monitoring model lifecycle.

-  **Notebook Examples**: Provides code in notebook format for user familiarity with the tool.

-  **Extensive Documentation, Unit Tests, and GitHub Actions Workflows**: Ensures *Sinergym* 
   is an efficient ecosystem for understanding and development.

-  And much more!

.. important:: If you want to introduce your own buildings in *Sinergym*, please visit :ref:`Adding new buildings for environments` section.

.. note:: *This is a work in progress project. Stay tuned for upcoming releases!*
