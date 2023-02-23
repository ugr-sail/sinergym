The goal of *sinergym* is to create an environment following *Gymnasium* interface for wrapping simulation engines (*Energyplus*) for building control using
**deep reinforcement learning**.

.. image:: /_static/operation_diagram.jpg
  :width: 800
  :alt: Sinergym diagram
  :align: center

|

.. note:: Please, help us to improve by **reporting your questions and issues** 
   `here <https://github.com/ugr-sail/sinergym/issues>`__. It is easy, just 2 clicks 
   using our issue templates (questions, bugs, improvements, etc.). More detailed 
   info on how to report issues 
   `here <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`__. 

The main functionalities of *Sinergym* are the following:

-  **Include different simulation engines**. Communication between
   Python and `EnergyPlus <https://energyplus.net/>`__ is established
   using `BCVTB <https://simulationresearch.lbl.gov/bcvtb/FrontPage>`__ middleware.
   Since this tool allows for interacting with several simulation
   engines, more of them (e.g.
   `OpenModelica <https://openmodelica.org/>`__) could be included in
   the backend while maintaining the Gymnasium API.

-  **Benchmark environments**. Similarly to *Atari* or *Mujoco* environments
   for RL community, we are designing a set of environments for
   benchmarking and testing deep RL algorithms. These environments may
   include different buildings, weathers, action/observation spaces, function rewards, etc.

-  **Customizable environments**. We aim to provide a
   package that allows to modify experimental settings in an easy
   manner. The user can create his own environments defining his own
   building model, weather, reward, observation/action space and variables, environment name, etc.
   The user can also use these pre-configured environments available in *Sinergym* 
   and change some aspect of it (for example, the weather) in such 
   a way that he does not  have to make an entire definition of the 
   environment and can start from one pre-designed by us.
   Some parameters directly associated with the simulator can be set as **extra configuration** 
   as well, such as people occupant, time-steps per simulation hour, run-period, etc.

-  **Customizable components**: *Sinergym* is easily scalable by third parties.
   Following the structure of the implemented classes, new custom components 
   can be created for new environments such as function rewards, wrappers,
   controllers, etc.

-  **Automatic Building Model adaptation to user changes**: Building models (*IDF*) will be
   adapted to specification of each simulation by the user. For example, ``Designdays`` and 
   ``Location`` components from *IDF* files will be adapted to weather file (*EPW*) specified in
   *Sinergym* simulator backend without any intervention by the user (only the environment definition).
   *BCVTB middleware* external interface in *IDF* model and *variables.cfg* file is generated when 
   simulation starts by *Sinergym*, this definition depends on action and observation space and variables defined.
   In short, *Sinergym* automates the whole process of model adaptation so that the user 
   only has to define what he wants for his environment.

-  **Automatic external interface integration for actions**. Sinergym provides functionality to obtain information 
   about the environments such as the zones or the schedulers available in the environment model. Using that information,
   which is possible to export in a excel, users can know which controllers are available in the building and, then, control 
   them with an external interface from an agent. To do this, users will make an **action definition** in which it is
   indicated which default controllers they want to replace in a specific format and *Sinergym* will take care of the relevant internal 
   changes in the model.

-  **Stable Baseline 3 Integration**. Some functionalities like callbacks
   have been customized by our team in order to test easily these environments
   with deep reinforcement learning algorithms. 
   This tool can be used with any other DRL library that supports the *Gymnasium* interface as well.

-  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with *Sinergym*, we tell you some details about how doing it.

-  **Mlflow tracking server**. `Mlflow <https://mlflow.org/>`__ is an open source platform for the machine
   learning lifecycle. This can be used with Google Cloud remote server (if you have Google Cloud account) 
   or using local store. This will help you to manage and store your runs and artifacts generated in an orderly
   manner.

-  **Data Visualization**. Using *Sinergym* logger or Tensorboard server to visualize training and evaluation information
   in real-time.

-  **Notebooks examples**. *Sinergym* develops code in notebook format with the purpose of offering use cases to 
   the users in order to help them become familiar with the tool. They are constantly updated, along with the updates 
   and improvements of the tool itself.

-  This project is accompanied by extensive **documentation**, **unit tests** and **github actions workflows** to make 
   *Sinergym* an efficient ecosystem for both understanding and development.

-  Many more!

.. note:: If you want to introduce your own buildings in Sinergym, please visit :ref:`Adding new buildings for environments` section.

.. note:: *This is a work in progress project. Stay tuned for upcoming releases!*
