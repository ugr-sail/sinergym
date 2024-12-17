#######################################
Deep Reinforcement Learning Integration
#######################################

*Sinergym* provides features to utilize **Deep Reinforcement Learning algorithms** from 
`Stable Baselines 3 <https://stable-baselines3.readthedocs.io/en/master/>`__. However, 
*Sinergym* is also compatible with any algorithm that operates with the Gymnasium interface.

To this end, we will refine and develop 
`Callbacks <https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html>`__, 
a set of functions that will be called at specific **stages of the training procedure**. 
Callbacks allow you to access the internal state of the RL model **during training**. 
They enable monitoring, auto-saving, model manipulation, progress bars, and more. 
Our callbacks inherit from Stable Baselines 3 and can be found in 
`sinergym/sinergym/utils/callbacks.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/callbacks.py>`__.

********************
Evaluation Callback
********************

An enhanced callback, named ``LoggerEvalCallback``, is used for evaluating the model versions obtained during 
the training process with *Sinergym*. It stores the best model obtained, not necessarily the final one from 
the training process. This callback inherits from Stable Baselines 3's ``EventCallback``. 

This callback functions similarly to EvalCallback from Stable Baselines 3 but includes numerous enhancements 
and specific adaptations for *Sinergym* and for logging relevant information during the process.

It requires that the evaluation environment be previously wrapped by a child class of ``BaseLoggerWrapper``. This is 
essential for the callback to access the logger's methods and attributes and to log information correctly.

Additionally, this callback saves the best model and evaluation summaries (CSV file) in a folder named ``evaluation`` 
within the training environment's output.

To log all this data to the Weights and Biases platform, the training environment must be previously wrapped 
with the ``WandbLoggerWrapper`` class (see :ref:`Logger Wrappers`). This allows evaluation data to be recorded on 
the Weights and Biases platform. Encapsulating the evaluation environment is not necessary unless detailed 
monitoring of these episodes is desired.

The data logged on the platform, in the Evaluations section, will depend on the specific logger wrapper used 
and its episode summary. Thus, to obtain new metrics, the logger wrapper must be modified, not the callback.

The number of episodes run in each evaluation and their frequency can be configured, and metrics from the 
underlying logger can be excluded if desired. Moreover, if normalization in observation space is applied,
the callback **will automatically copy the calibration parameters** from the training environment to the evaluation
environment.

More episodes lead to more accurate averages of the reward-based indicators, providing a more realistic 
assessment of the current model's performance. However, this increases the time required. To see an example,
visit :ref:`Training a model`.

************
Usage
************

Model Training
~~~~~~~~~~~~~~~~

Leverage this functionality for your experiments using the script 
`sinergym/scripts/train/train_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/train_agent.py>`__.

Key considerations for your experiments:

* Models are built using an algorithm constructor, each with its own **specific parameters**. 
  Defaults are used if none are defined.

* If you normalize the environment wrapper, models will **train** in these **normalized** spaces.

* **Concatenate** callbacks in a ``CallbackList`` instance from Stable Baselines 3.

* The neural network begins training upon executing the ``model.learn()`` method, where you specify ``timesteps``, 
  ``callbacks``, and ``log_interval``.

* **Curriculum Learning** can be implemented by adding a model field with a valid model path. 
  The script will load and train the model.

The ``train_agent.py`` script requires a single parameter, ``-conf``, a string indicating the JSON 
file containing all experiment details. Refer to the JSON structure in `sinergym/scripts/train/train_agent_PPO.json <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/train_agent_PPO.json>`__:

* **Mandatory** parameters: environment, episodes, algorithm (and any non-default algorithm parameters).

* **Optional** parameters: environment parameters (overwrites default if specified), seed, pre-training 
  model to load, experiment ID, wrappers (in order), training evaluation, and cloud options.

* Field names must match the example, or the experiment will fail.

The script performs the following:

    1. Names the experiment in the format: ``<algorithm>-<environment_name>-episodes<episodes_int>-seed<seed_value>(<experiment_date>)``

    2. Sets environment parameters if specified.

    3. Applies specified wrappers from JSON.

    4. Save all experiment's hyperparameters in WandB if a session is detected.

    5. Defines model algorithm with specified hyperparameters.

    6. Calculates training timesteps from number of episodes.

    7. Sets up evaluation callback if specified.

    8. Trains with environment.

    9. If remote store is specified, saves all outputs in Google Cloud Bucket. If wandb is specified, saves all outputs in wandb run artifact.

    10. Auto-deletes remote container in Google Cloud Platform if auto-delete parameter is specified.


Model Loading
~~~~~~~~~~~~~~~~~~~~~~

Use the script `sinergym/scripts/eval/load_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/eval/load_agent.py>`__ 
to load and evaluate or execute a previously trained model.

The ``load_agent.py`` script requires a single parameter, ``-conf``, a string indicating the JSON file 
containing all evaluation details. Refer to the JSON structure in 
`sinergym/scripts/eval/load_agent_example.json <https://github.com/ugr-sail/sinergym/blob/main/scripts/eval/load_agent_example.json>`__:

* **Mandatory** parameters: environment, episodes, algorithm (name only), and model to load: local (``model``) or remote (``wandb_model``).

* **Optional** parameters: environment parameters (overwrites default if specified), experiment ID, 
  wrappers (in order), and cloud options.

The script loads the model and predicts actions from states over the specified episodes. The information 
is collected and sent to remote storage (like WandB) if specified, otherwise it remains in local memory.

The model field in JSON can be a **local path** to the model, a **bucket url** in the form 
``gs://``, or a *wandb* artifact path for stored models.

.. note:: *This project is a work in progress. Direct support for additional algorithms is planned for the future!*