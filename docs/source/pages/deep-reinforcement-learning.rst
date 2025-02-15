#######################################
Deep Reinforcement Learning integration
#######################################

*Sinergym* is compatible with any controller that operates under the Gymnasium interface, and can be used with most existing **Deep Reinforcement Learning** (DRL) libraries.

It has a close integration with `Stable Baselines 3 <https://stable-baselines3.readthedocs.io/en/master/>`__, especially regarding the use of **callbacks**.  Callbacks are functions called at specific stages of DRL agents execution. They allow access to the internal state of the DRL model during training, enabling monitoring, auto-saving, model manipulation, progress visualization, and more.  

Pre-implemented callbacks provided by *Sinergym* inherit from Stable Baselines 3 and can be found in `sinergym/sinergym/utils/callbacks.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/callbacks.py>`__.

******************
LoggerEvalCallback
******************

The ``LoggerEvalCallback`` is used to evaluate the different model versions obtained during the training process of the agent. It saves the best model obtained, not necessarily the final one from the training process. This callback inherits from the ``EventCallback`` of Stable Baselines 3.

This callback is similar to the ``EvalCallback`` of Stable Baselines 3 but includes numerous enhancements and specific adaptations for *Sinergym*, in particular for logging relevant simulation data during the training process.

The evaluation environment must be first wrapped by a child class of ``BaseLoggerWrapper``. This is essential for the callback to access the logger's methods and attributes, and to log the information correctly.

In addition, this callback stores the best model and evaluation summaries (in CSV format) in a folder named ``evaluation`` within the training environment output.

Weights And Biases logging
~~~~~~~~~~~~~~~~~~~~~~~~~~

To log all this data to the `Weights and Biases <https://wandb.ai/>`__ platform, the training environment must be first wrapped with the ``WandbLoggerWrapper`` class (see :ref:`Logger Wrappers`). Encapsulation of the evaluation environment is not necessary unless detailed monitoring of these episodes is desired.

The data logged to the platform (in the *Evaluations* section) depends on the specific logger wrapper used and its episode summary. Therefore, to get new metrics, the logger wrapper must be modified, not the callback. In addition, this callback will overwrite certain metrics for the best model obtained during the training process, in order to preserve the metrics of the best model.

The number of episodes run in each evaluation and their frequency can be configured, and metrics from the underlying logger can be excluded if desired. Moreover, if the observation space is normalized, the callback **automatically copies the calibration parameters** from the training environment to the evaluation environment.

More episodes lead to more accurate averages of the reward-based indicators, providing a more realistic assessment of the current model's performance. However, this will increase the time required. For a detailed usage example, see :ref:`Training a model`.

*****
Usage
*****

Model training
~~~~~~~~~~~~~~

If you are looking to train a DRL agent using *Sinergym*, we provide the script `sinergym/scripts/train/local_confs/train_agent_local_conf.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/local_confs/train_agent_local_conf.py>`__. which can be easily adapted to custom experiments.

The following are some key points to consider:

* Models are built using an algorithm constructor, each with its own **specific parameters**. Defaults are used if none are defined.

* If you normalize the environment wrapper, models will **train** using these **normalized** spaces.

* Callbacks are **concatenated** by using a ``CallbackList`` instance from Stable Baselines 3.

* The model begins training once the ``model.learn()`` method is called. The parameters ``timesteps``, 
  ``callbacks``, and ``log_interval`` are specified there.

* **Sequential / curriculum learning** can be implemented by adding a valid model path to the ``model`` parameter. In this way, the script will load and re-train an existing model.

The ``train_agent_local_conf.py`` script requires a single parameter (``-conf``), which is the YAML file containing the experiment configuration. A sample YAML structure with comments to understand the structure is detailed in `sinergym/scripts/train/local_confs/conf_examples/train_agent_PPO.yaml <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/local_confs/conf_examplestrain_agent_PPO.yaml>`__.

We distinguish between *mandatory* and *optional* parameters:

* **Mandatory**: environment, training episodes, and algorithm (plus any non-default algorithm parameters).

* **Optional**: environment parameters (overwrites default if specified), seed, pre-training 
  model to load, experiment ID, wrappers (in order), training evaluation, and cloud options.

Once executed, the script performs the following steps:

  1. Names the experiment following the format: ``<algorithm>-<environment_name>-episodes<episodes>-seed<seed_value>(<experiment_date>)``.

  2. Sets environment parameters if specified.

  3. Applies specified wrappers from the YAML configuration.

  4. Saves all experiment's hyperparameters in WandB if a session is detected.

  5. Defines the model algorithm with the specified hyperparameters. If a model has been specified, loads it and continues training.

  6. Calculates training timesteps from the number of episodes.

  7. Sets up an evaluation callback if specified.

  8. Trains the model with the environment.

  9. If a remote store is specified, saves all outputs in a Google Cloud Bucket. If WandB is specified, saves all outputs in the WandB run artifact.

  10. Auto-deletes the remote container in Google Cloud Platform if the auto-delete parameter is specified.

Model training with sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Weights and Biases sweeps <https://docs.wandb.ai/guides/sweeps/>`__ is a powerful feature that enables hyperparameter exploration in artificial intelligence algorithms.

To help users take advantage of this functionality, we have created a script that allows agents to run in parallel or sequentially. These agents pick predefined configurations from previously created sweeps to carry out the optimization process.

The script for launching agents, the training script they execute (either in parallel or sequentially), and example sweep configurations can all be found in the `sinergym/scripts/train/sweep_confs <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/sweep_confs>`__ directory.

We recommend reviewing the contents of this directory alongside the Weights and Biases documentation if you are interested in using this feature.

Model loading
~~~~~~~~~~~~~~~~~~~~~~

To load and evaluate/execute an previously trained model, use the script `sinergym/scripts/eval/load_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/eval/load_agent.py>`__. 

The ``load_agent.py`` script requires a single parameter, ``-conf``, indicating the YAML file with the evaluation configuration. See the YAML structure in 
`sinergym/scripts/eval/load_agent_example.yaml <https://github.com/ugr-sail/sinergym/blob/main/scripts/eval/load_agent_example.yaml>`__ for a reference example of this configuration file.

Again, we distinguish between *mandatory* and *optional* parameters:

* **Mandatory**: environment, evaluation episodes, algorithm (name only), and model to load. The model field can be a *local path*, a *bucket url* in the form ``gs://``, or a WandB artifact path for stored models.

* **Optional**: environment parameters (which overwrite defaults if specified), experiment identifier, wrappers (in order), and cloud options.

The script loads the model and executes it the specified environment. Relevant data is collected and sent to remote storage if specified, otherwise it is stored locally.