.. seealso:: This is a project based on Zhiang Zhang and Khee Poh Lam `Gym-Eplus <https://github.com/zhangzhizza/Gym-Eplus>`__.

The goal of *sinergym* is to create an environment following OpenAI
Gym interface for wrapping simulation engines for building control using
**deep reinforcement learning**.

.. image:: /_static/operation_diagram.jpg
  :width: 800
  :alt: Sinergym diagram
  :align: center

The main functionalities of *sinergym* are the following:

-  **Benchmark environments**. Similarly to Atari or Mujoco environments
   for RL community, we are designing a set of environments for
   benchmarking and testing deep RL algorithms. These environments may
   include different buildings, weathers or action spaces.
-  **Develop different experimental settings**. We aim to provide a
   package that allows to modify experimental settings in an easy
   manner. For example, several reward functions or observation
   variables may be defined.
-  **Include different simulation engines**. Communication between
   Python and `EnergyPlus <https://energyplus.net/>`__ is established
   using `BCVTB <https://simulationresearch.lbl.gov/bcvtb/FrontPage>`__.
   Since this tool allows for interacting with several simulation
   engines, more of them (e.g.
   `OpenModelica <https://openmodelica.org/>`__) could be included in
   the backend while maintaining the Gym API.
-  **Stable Baseline 3 Integration**. Some functionalities like callbacks
   have been developed by our team in order to test easily these environments
   with deep reinforcement learning algorithms.
-  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with Sinergym, it has been designed a complete functionality
   in order to facilitate this work.
-  **Data Visualization**. Using Sinergym logger or Tensorboard server to visualize training information
   in real-time.
-  Many more!

.. note:: *This is a work in progress project. Stay tuned for upcoming releases!*