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

If you want to train a DRL agent using *Sinergym*, you can use the script `sinergym/scripts/train/local_confs/train_agent_local_conf.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/local_confs/train_agent_local_conf.py>`__, which is easily adaptable for custom experiments.

Here are a few key points to consider:

* Models are instantiated using an algorithm constructor, each with its own **specific parameters**. Defaults are used if none are provided.

* If you apply a normalization wrapper to the environment, models will **train** using these **normalized** spaces.

* Callbacks are **combined** using a ``CallbackList`` from Stable Baselines3.

* Training starts when the ``model.learn()`` method is called. Important parameters such as ``total_timesteps``, ``callback``, and ``log_interval`` are passed here.

* **Sequential / curriculum learning** is supported by providing a path to a previously trained model using the ``model`` parameter. This allows resuming or fine-tuning a model.

The ``train_agent_local_conf.py`` script requires a single argument (``-conf``), which should point to a YAML configuration file. An example configuration file with detailed comments can be found here: `train_agent_PPO.yaml <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/local_confs/conf_examples/train_agent_PPO.yaml>`__.

We distinguish between *mandatory* and *optional* configuration parameters:

* **Mandatory**: environment, number of training episodes, and algorithm (including non-default hyperparameters if needed).

* **Optional**: environment parameters (override defaults), random seed, pretrained model path, experiment ID, wrappers (in order), evaluation settings, and cloud integration options.

Once executed, the script performs the following steps:

1. **Generate the experiment name** using the format ``<experiment_name>_<date>`` if ``experiment_name`` is specified, or ``<algorithm_name>_<date>`` otherwise.

2. **Load a pretrained model**, if defined in the configuration:

   - From a local file path.
   - From a Weights & Biases (WandB) artifact.
   - From a Google Cloud Storage bucket.

3. **Load and configure environment parameters**:

   - If an environment YAML configuration is provided, load all parameters from it (:ref:`Environment Configuration Serialization`).
   - Optionally override or extend specific parameters using ``env_params`` in the configuration.
   - Set the ``env_name`` to match the experiment name for better traceability.

4. **Apply wrappers to the environment**, if specified:

   - Load wrapper settings from a YAML file (:ref:`Wrapper Serialization and Restoration`).
   - Optionally override or add wrappers defined directly in the configuration.
   - Supports custom objects or callables using the ``<module>:<object>`` format.

5. **Create the simulation environment**, applying all parameters and wrappers.

6. **Log experiment metadata to Weights & Biases**, if ``WandBLogger`` is active:

   - Track Sinergym, Python, and Stable-Baselines3 versions.
   - Store the full configuration and the processed environment parameters.

7. **Initialize the RL algorithm** using the specified hyperparameters:

   - If no model is loaded, training starts from scratch. Using the algorithm hyperparameters defined in the configuration.
   - If a pretrained model is available, it resumes training from the saved state.

8. **Set up custom logging**, combining console and WandB logging when ``WandBLogger`` is enabled.

9. **Prepare evaluation**, if enabled:

   - Create a separate evaluation environment (excluding ``WandBLogger``).
   - Set up a ``LoggerEvalCallback`` to run periodic evaluations during training.

10. **Calculate total training timesteps** based on the number of episodes and episode length.

11. **Train the model** using the environment and configured callbacks.

12. **Save the final model** in the environment’s ``workspace_path`` after training completes.

13. **Handle errors and interruptions gracefully**:

    - Save the model state.
    - Close the environment properly.
  
.. important:: The YAML configuration structure and values are designed to be **intuitive and easy to use**, especially when 
   paired with this documentation. To get started, simply explore one of the provided example configuration files.  
   These examples clearly illustrate how to define your environment, wrappers, algorithm, and other training 
   options—making it straightforward to set up your own experiments. Visit `sinergym/scripts/train/local_confs/conf_examples <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/local_confs/conf_examples>`__.

.. warning:: If you are loading a pretrained model that was trained with **observation normalization**,  
   it is **critical** to also load the **normalization statistics** (i.e., the running mean and variance)  
   used during its original training (see :ref:`NormalizeObservation`). Otherwise, the model may perform poorly or behave unpredictably due  
   to mismatched input distributions. These statistics are typically saved along with the model and should  
   be restored explicitly before continuing training or evaluation, setting up the NormalizeObservation wrapper.

Model training with sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Weights and Biases sweeps <https://docs.wandb.ai/guides/sweeps/>`__ is a powerful feature that enables hyperparameter exploration in artificial intelligence algorithms.

To help users take advantage of this functionality, we have created a script that allows agents to run in parallel or sequentially. These agents pick predefined configurations from previously created sweeps to carry out the optimization process. The process is similar to the one described in the previous section.

The script for launching agents, the training script they execute (either in parallel or sequentially), and example sweep configurations can all be found in the `sinergym/scripts/train/sweep_confs <https://github.com/ugr-sail/sinergym/blob/main/scripts/train/sweep_confs>`__ directory.

We recommend reviewing the contents of this directory alongside the Weights and Biases documentation if you are interested in using this feature.

Model loading
~~~~~~~~~~~~~

To load and evaluate a previously trained model, you can use the script `scripts/eval/load_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/eval/load_agent.py>`__. This script is flexible and supports multiple model sources and environment configurations.

The script requires a single parameter, ``-conf``, pointing to a YAML file with the evaluation setup. A reference configuration can be found in `scripts/eval/load_agent_example.yaml <https://github.com/ugr-sail/sinergym/blob/main/scripts/eval/load_agent_example.yaml>`__.

We distinguish between *mandatory* and *optional* parameters:

* **Mandatory**: environment name, number of episodes, algorithm (only name is required), and model path. Supported model sources include:
  
  - Local file path
  - Google Cloud Storage bucket (``gs://...`` format)
  - Weights & Biases (WandB) artifact

* **Optional**: environment parameters (overrides defaults if provided), experiment name, wrapper definitions, and cloud storage options.

During execution, the script performs the following steps:

1. Generates a unique evaluation name (e.g., ``PPO_2025-05-29_10:12_evaluation``).
2. Downloads and loads the specified model from the defined source.
3. Loads environment and wrapper configurations (from YAML environment and wrappers serialization or directly from the config).
4. Initializes the evaluation environment with all parameters and wrappers.
5. Runs the agent for the defined number of episodes.
6. Stores results locally or in the cloud, depending on configuration.
7. Gracefully handles errors and interruptions, ensuring environment closure.

.. warning::

   If your model was trained with **observation normalization**, make sure to restore the corresponding  
   **normalization statistics**. These are usually saved with the model and must be loaded to ensure  
   the agent receives inputs with the expected distribution. See :ref:`NormalizeObservation` for more details on how to handle this.