#######################################
Deep Reinforcement Learning Integration
#######################################

*Sinergym* integrates some facilities in order to use **Deep Reinforcement Learning algorithms** 
provided by `Stable Baselines 3 <https://stable-baselines3.readthedocs.io/en/master/>`__. 
Current algorithms checked by *Sinergym* are:

+--------------------------------------------------------+
|                   Stable Baselines 3:                  |
+-----------+----------+------------+--------------------+
| Algorithm | Discrete | Continuous |        Type        |
+-----------+----------+------------+--------------------+
| PPO       |    YES   |     YES    | OnPolicyAlgorithm  |
+-----------+----------+------------+--------------------+
| A2C       |    YES   |     YES    | OnPolicyAlgorithm  |
+-----------+----------+------------+--------------------+
| DQN       |    YES   |     NO     | OffPolicyAlgorithm |
+-----------+----------+------------+--------------------+
| DDPG      |    NO    |     YES    | OffPolicyAlgorithm |
+-----------+----------+------------+--------------------+
| SAC       |    NO    |     YES    | OffPolicyAlgorithm |
+-----------+----------+------------+--------------------+
| TD3       |    NO    |     YES    | OffPolicyAlgorithm |
+-----------+----------+------------+--------------------+

``Type`` column has been specified due to its importance about 
*Stable Baselines callback* functionality.

****************
DRL Logger
****************

`Callbacks <https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html>`__ 
are a set of functions that will be called at given **stages of the training procedure**. 
You can use callbacks to access internal state of the RL model **during training**. 
It allows one to do monitoring, auto saving, model manipulation, progress bars, ...

This structure allows to custom our own logger for DRL executions. Our objective 
is to **log all information about our custom environment** specifically.
Therefore, `sinergym/sinergym/utils/callbacks.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/callbacks.py>`__ 
has been created with this proposal. Each algorithm has its own differences 
about how information is extracted which is why its implementation. ``LoggerCallback``
can deal with those subtleties.

.. note:: You can specify if you want Sinergym logger (see :ref:`Logger`) to record 
          simulation interactions during training at the same time using 
          ``sinergym_logger`` attribute in constructor. 

This callback derives ``BaseCallback`` from Stable Baselines 3 while ``BaseCallBack`` 
uses `Tensorboard <https://www.tensorflow.org/tensorboard?hl=es-419>`__ on the 
background at the same time. With *Tensorboard*, it's possible to visualize all DRL 
training in real time and compare between different executions. This is an example: 

.. image:: /_static/tensorboard_example.png
  :width: 800
  :alt: Tensorboard example
  :align: center

There are tables which are in some algorithms and not in others and vice versa. 
It is important the difference between ``OnPolicyAlgorithms`` and ``OffPolicyAlgorithms``:

* **OnPolicyAlgorithms** can be recorded **each timestep**, we can set a ``log_interval`` in 
  learn process in order to specify the **step frequency log**.

* **OffPolicyAlgorithms** can be recorded **each episode**. Consequently, ``log_interval`` in 
  learn process is used to specify the **episode frequency log** and not step frequency.
  Some features like actions and observations are set up in each timestep. 
  Thus, Off Policy Algorithms record a **mean value** of whole episode values instead 
  of values steps by steps (see ``LoggerCallback`` class implementation).

~~~~~~~~~~~~~~~~~~~~~~
Tensorboard structure
~~~~~~~~~~~~~~~~~~~~~~

The main structure for *Sinergym* with *Tensorboard* is:

* **action**: This section has action values during training. When algorithm 
  is On Policy, it will appear **action_simulation** too. This is because 
  algorithms in continuous environments has their own output and clipped 
  with gym action space. Then, this output is parse to simulation action 
  space (See :ref:`Observation/action spaces` note box).

* **episode**: Here is stored all information about entire episodes. 
  It is equivalent to ``progress.csv`` in *Sinergym logger* 
  (see *Sinergym* :ref:`Output format` section):

    - *comfort_violation_time(%)*: Percentage of time in episode simulation 
      in which temperature has been out of bound comfort temperature ranges.

    - *cumulative_comfort_penalty*: Sum of comfort penalties (reward component) 
      during whole episode.

    - *cumulative_power*: Sum of power consumption during whole episode.

    - *cumulative_power_penalty*: Sum of power penalties (reward component) 
      during whole episode.

    - *cumulative_reward*: Sum of reward during whole episode.

    - *ep_length*: Timesteps executed in each episode.

    - *mean_comfort_penalty*: Mean comfort penalty per step in episode.

    - *mean_power*: Mean power consumption per step in episode.

    - *mean_power_penalty*: Mean power penalty per step in episode.

    - *mean_reward*: Mean reward obtained per step in episode.

* **observation**: Here is recorded all observation values during simulation. 
  This values depends on the environment which is being simulated 
  (see :ref:`Observation/action spaces` section).

* **normalized_observation** (optional): This section appear only when environment 
  has been **wrapped with normalization** (see :ref:`Wrappers` section). The model 
  will train with this normalized values and they will be recorded both; 
  original observation and normalized observation.

* **rollout**: Algorithm metrics in **Stable Baselines by default**. For example, 
  DQN has ``exploration_rate`` and this value doesn't appear in other algorithms.

* **time**: Monitoring time of execution.

* **train**: Record specific neural network information for each algorithm, 
  provided by **Stable baselines** as well as rollout.

.. note:: Evaluation of models can be recorded too, adding ``EvalLoggerCallback`` 
          to model learn method.

**********
How use
**********

You can try your own experiments and benefit from this functionality. 
`sinergym/scripts/DRL_battery.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/DRL_battery.py>`__
is a example code to use it. You can use ``DRL_battery.py`` directly from 
your local computer specifying ``--tensorboard`` flag in execution.

The most **important information** you must keep in mind when you try 
your own experiments are:

* Model is constructed with a algorithm constructor. 
  Each algorithm can use its **particular parameters**.

* If you wrapper environment with normalization, models 
  will **train** with those **normalized** values.

* Callbacks can be **concatenated** in a ``CallbackList`` 
  instance from Stable Baselines 3.

* Neural network will not train until you execute 
  ``model.learn()`` method. Here is where you
  specify train ``timesteps``, ``callbacks`` and ``log_interval`` 
  as we commented in type algorithms (On and Off Policy).

* ``DRL_battery.py`` requires some **extra arguments** to being 
  executed like ``-env`` and ``-ep``.

* You can execute **Curriculum Learning**, you only have to 
  add ``--model`` field with a valid model path, this script 
  will load the model and execute to train.

****************
Mlflow
****************

Our scripts to run DRL with *Sinergym* environments are using
`Mlflow <https://mlflow.org/>`__ in order to **tracking experiments** 
and recorded them methodically. It is recommended to use it.
You can start a local server with information stored during the 
battery of experiments such as initial and ending date of execution, 
hyperparameters, duration, etc.

Here is an example: 

.. image:: /_static/mlflow_example.png
  :width: 800
  :alt: Tensorboard example
  :align: center


.. note:: For information about how use *Tensorboard* and *Mlflow* with a Cloud 
          Computing paradigm, see :ref:`Remote Tensorboard log` and 
          :ref:`Mlflow tracking server set up`.

.. note:: *This is a work in progress project. Direct support with others 
          algorithms is being planned for the future!*