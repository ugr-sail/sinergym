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
   include different buildings, weathers or action/observation spaces.
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
-  **Building Models configuration automatically**: Building models will be
   adapted to specification of each simulation. For example, *Designdays* and 
   *Location* from IDF files will be adapted to weather file specified in
   Sinergym simulator backend without any intervention by the user.
-  **Extra configuration facilities**: Our team aim to provide extra parameters
   in order to amplify the context space for the experiments with this tool.
   Sinergym will modify building model automatically based on parameters set.
   For example: People occupant, timesteps per simulation hour, observation
   and action spaces, etc.
-  **Stable Baseline 3 Integration**. Some functionalities like callbacks
   have been developed by our team in order to test easily these environments
   with deep reinforcement learning algorithms.
-  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with Sinergym, we tell you some details about how doing it.
-  **Mlflow tracking server**. `Mlflow <https://mlflow.org/>`__ is an open source platform for the machine
   learning lifecycle. This can be used with Google Cloud remote server (if you have Google Cloud account) 
   or using local store. This will help you to manage and store your runs and artifacts generated in an orderly
   manner.
-  **Data Visualization**. Using Sinergym logger or Tensorboard server to visualize training information
   in real-time.
-  Many more!

.. note:: *This is a work in progress project. Stay tuned for upcoming releases!*