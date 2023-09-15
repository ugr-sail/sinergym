#######################################
Deep Reinforcement Learning Integration
#######################################

*Sinergym* integrates some facilities in order to use **Deep Reinforcement Learning algorithms** 
provided by `Stable Baselines 3 <https://stable-baselines3.readthedocs.io/en/master/>`__. Although *Sinergym* is 
compatible with any algorithm which works with Gymnasium interface.

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

For that purpose, we are going to refine and develop 
`Callbacks <https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html>`__ 
which are a set of functions that will be called at given **stages of the training procedure**. 
You can use callbacks to access internal state of the RL model **during training**. 
It allows one to do monitoring, auto saving, model manipulation, progress bars, ...
Our callbacks inherit from Stable Baselines 3 and are available in 
`sinergym/sinergym/utils/callbacks.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/callbacks.py>`__.

``Type`` column has been specified due to its importance about 
*Stable Baselines callback* functionality.

********************
DRL Callback Logger
********************

A callback allows to custom our own logger for DRL *Sinergym* executions. Our objective 
is to **log all information about our custom environment** specifically in real-time.
Each algorithm has its own differences 
about how information is extracted which is why its implementation.

.. note:: You can specify if you want *Sinergym* logger (see :ref:`Logger`) to record 
          simulation interactions during training at the same time using 
          ``sinergym_logger`` attribute in constructor. 

The ``LoggerCallback`` inherits from Stable Baselines 3 ``BaseCallback`` and 
uses `Weights&Biases <https://wandb.ai/site>`__ (*wandb*) in the background in order to host 
all information extracted. With *wandb*, it's possible to track and visualize all DRL 
training in real time, register hyperparameters and details of each execution, save artifacts 
such as models and *Sinergym* output, and compare between different executions. This is an example: 

- Hyperparameter and summary registration:

.. image:: /_static/wandb_example1.png
  :width: 800
  :alt: WandB hyperparameters
  :align: center

- Artifacts registered (if evaluation is enabled, best model is registered too):

.. image:: /_static/wandb_example2.png
  :width: 800
  :alt: WandB artifacts
  :align: center

- Metrics visualization in real time:

.. image:: /_static/wandb_example3.png
  :width: 800
  :alt: WandB charts
  :align: center

There are tables which are in some algorithms and not in others and vice versa. This depends on the algorithm used.

.. note:: Since version 3.0.6, *Sinergym* can record real-time data with the same timestep frequency regardless of the algorithm. See `#363 <https://github.com/ugr-sail/sinergym/pull/363>`__.

********************
Evaluation Callback
********************

A callback has also been refined for the evaluation of the model versions obtained during 
the training process with *Sinergym*, so that it stores the best model obtained (not the one resulting 
at the end of the training).

Its name is ``LoggerEvalCallback`` and it inherits from Stable Baselines 3 ``EventCallback``. 
The main feature added is that the model evaluation is logged in a particular section in 
*wandb* too for the concrete metrics of the building model. The evaluation is customized for
*Sinergym* particularities. 

We have to define in ``LoggerEvalCallback`` construction how many training episodes we want 
the evaluation process to take place. On the other hand, we have to define how many episodes 
are going to occupy each of the evaluations to be performed. 

With more episodes, more accurate the averages of the reward-based indicators will be, and, 
therefore, the more faithful it will be to reality in terms of how good the current model is 
turning out to be. However, it will take more time.

It calculates timestep and episode averages for power consumption, temperature violation, comfort penalty and power penalty.
On the other hand, it calculates comfort violation percentage in episodes too.
Currently, only mean reward is taken into account to decide when a model is better.

******************************
Weights and Biases structure
******************************

The main structure for *Sinergym* with *wandb* is:

* **action_network**: The raw output returned by the network in DRL algorithm.

* **action_simulation**: The transformed output, being the values that the simulator 
  takes for processing and calculation of the next state and reward in Sinergym.

* **episode**: Here is stored all information about entire episodes. 
  It is equivalent to ``progress.csv`` in *Sinergym logger* 
  (see *Sinergym* :ref:`Output format` section):

    - *ep_length*: Timesteps executed in each episode.

    - *cumulative_reward*: Sum of reward during whole episode.

    - *mean_reward*: Mean reward obtained per step in episode.

    - *cumulative_power*: Sum of power consumption during whole episode.

    - *mean_power*: Mean of power consumption per step during whole episode.

    - *cumulative_temperature_violation*: Sum of temperature (Cº) out of comfort range during whole episode.

    - *mean_temperature_violation*: Mean of temperature (Cº) out of comfort range per step during whole episode.

    - *cumulative_comfort_penalty*: Sum of comfort penalties (reward component) 
      during whole episode.

    - *mean_comfort_penalty*: Mean of comfort penalties per step (reward component) 
      during whole episode.

    - *cumulative_energy_penalty*: Sum of energy penalties (reward component) 
      during whole episode.

    - *mean_energy_penalty*: Mean of energy penalties per step (reward component) 
      during whole episode.

    - *comfort_violation_time(%)*: Percentage of time in episode simulation 
      in which temperature has been out of bound comfort temperature ranges.

* **observation**: Here is recorded all observation values during simulation. 
  This values depends on the environment which is being simulated 
  (see :ref:`action space` section).

* **normalized_observation** (optional): This section appear only when environment 
  has been **wrapped with normalization** (see :ref:`Wrappers` section). The model 
  will train with this normalized values and they will be recorded both; 
  original observation and normalized observation.

* **rollout**: Algorithm metrics in **Stable Baselines by default**. For example, 
  DQN has ``exploration_rate`` and this value doesn't appear in other algorithms.

* **time**: Monitoring time of execution.

* **train**: Record specific neural network information for each algorithm, 
  provided by **Stable baselines** as well as rollout.

* **eval**: Record all evaluations done during training if the callback has been set up.
  The graphs here are the same than in *episode* label.

.. note:: Evaluation of models can be recorded too, adding ``EvalLoggerCallback`` 
          to model learn method.

.. note:: For more information about how to use it with cloud computing, visit :ref:`Sinergym with Google Cloud`.

************
How to use
************

Train a model
~~~~~~~~~~~~~~~~

You can try your own experiments and benefit from this functionality. 
`sinergym/scripts/train_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/train_agent.py>`__
is a script to help you to do it.

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

* You can execute **Curriculum Learning**, you only have to 
  add model field with a valid model path, this script 
  will load the model and execute to train.

``train_agent.py`` has a unique parameter to be able to execute it; ``-conf``.
This parameter is a str to indicate the JSON file in which there are allocated
all information about the experiment you want to execute. You can see the
JSON structure example in `sinergym/scripts/train_agent_example.json <https://github.com/ugr-sail/sinergym/blob/main/scripts/train_agent_example.json>`__:

* The **obligatory** parameters are: environment, episodes, 
  algorithm (and parameters of the algorithm which don't have 
  default values).

* The **optional** parameters are: All environment parameters (if it is specified 
  will be overwrite the default environment value), seed, model to load (before training),
  experiment ID, wrappers to use (respecting the order), training evaluation,
  wandb functionality and cloud options.

* The name of the fields must be like in example mentioned. Otherwise, the experiment
  will return an error.

This script do the next:

    1. Setting an appropriate name for the experiment. Following the next
       format: ``<algorithm>-<environment_name>-episodes<episodes_int>-seed<seed_value>(<experiment_date>)``

    2. Starting WandB track experiment with that name (if configured in JSON), it will create an local path (*./wandb*) too.

    3. Log all parameters allocated in JSON configuration (including *sinergym.__version__* and python version).

    4. Setting env with parameters overwritten in case of establishing them.

    5. Setting wrappers specified in JSON.

    6. Defining model algorithm using hyperparameters defined.

    7. Calculate training timesteps using number of episodes.

    8. Setting up evaluation callback if it has been specified.

    9. Setting up WandB logger callback if it has been specified.

    10. Training with environment.

    11. If remote store has been specified, saving all outputs in Google 
        Cloud Bucket. If wandb has been specified, saving all 
        outputs in wandb run artifact.

    12. Auto-delete remote container in Google Cloud Platform when parameter 
        auto-delete has been specified.


Load a trained model
~~~~~~~~~~~~~~~~~~~~~~

You can try to load a previous trained model and evaluate or execute it. 
`sinergym/scripts/load_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/load_agent.py>`__
is a script to help you to do it.

``load_agent.py`` has a unique parameter to be able to execute it; ``-conf``.
This parameter is a str to indicate the JSON file in which there are allocated
all information about the evaluation you want to execute. You can see the
JSON structure example in `sinergym/scripts/load_agent_example.json <https://github.com/ugr-sail/sinergym/blob/main/scripts/load_agent_example.json>`__:

* The **obligatory** parameters are: environment, episodes,
  algorithm (only algorithm name is necessary) and model to load.

* The **optional** parameters are: All environment parameters (if it is specified 
  will be overwrite the default environment value),
  experiment ID, wrappers to use (respecting the order),
  wandb functionality and cloud options.

This script loads the model. Once the model is loaded, it predicts the actions from the 
states during the agreed episodes. The information is collected and sent to a remote
storage if it is indicated (such as WandB), 
otherwise it is stored in local memory only.

The model field in JSON can be a **local path** with the model, a **bucket url** with the form ``gs://``,
or a *wandb* artifact path if we have some model stored there.

.. note:: *This is a work in progress project. Direct support with others 
          algorithms is being planned for the future!*