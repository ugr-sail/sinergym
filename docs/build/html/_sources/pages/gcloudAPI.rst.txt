#########################
Sinergym Google Cloud API 
#########################

In this project, an API based on RESTfull API for gcloud has been designed and developed in order to use Google Cloud infrastructure directly writing experiments definition ir our personal computer.

.. image:: /_static/Sinergym_cloud_API.png
  :width: 1000
  :alt: Sinergym cloud API diagram
  :align: center

Let's see a detailed explanation of the diagram above.

****************
Executing API
****************

Our objective is defining a set of experiments in order to execute them in a Google Cloud remote container each one. For this, *cloud_manager.py* has been created in repository root. This file must be used in our local computer:

.. literalinclude:: ../../../cloud_manager.py
    :language: python

This script uses the following parameters:

- ``--project_id`` or ``-id``: Your Google Cloud project id must be specified.
- ``--zone`` or ``-zo``: Zone for your project (default is *europe-west1-b*).
- ``--template_name`` or ``-tem``: Template used to generate VM's clones, defined in your project previously (see :ref:`4. Create your VM or MIG`).
- ``--group_name`` or ``-group``: Instance group name you want. All instances inner MIG will have this name concatenated with a random str.
- ``--experiment_commands`` or ``-cmds``: Experiment definitions list using python command format (for information about its format, see :ref:`Receiving experiments in remote containers`).

Here is an example bash code to execute the script:

.. code:: sh

    $ python cloud_manager.py \
        --project_id sinergym \
        --zone europe-west1-b \
        --template_name sinergym-template \
        --group_name sinergym-prueba2 \
        --experiment_commands \
        'python3 DRL_battery.py --environment Eplus-5Zone-hot-discrete-v1 --episodes 2 --algorithm DQN --logger --log_interval 1 --seed 58 --evaluation --eval_freq 1 --eval_length 1 --tensorboard ./tensorboard_log --remote_store' \
        'python3 DRL_battery.py --environment Eplus-5Zone-hot-continuous-v1 --episodes 3 --algorithm PPO --logger --log_interval 300 --seed 52 --evaluation --eval_freq 1 --eval_length 1 --tensorboard ./tensorboard_log --remote_store'

This example generates only 2 machines inner an instance group in your Google Cloud Platform because of you have defined two experiments. If you defined more experiments, more machines will be created by API.

.. note:: Because of its real-time process. Some containers, instance list action and others could take time. In that case, the API wait a process finish to execute the next (when it is necessary).

.. note:: This script uses gcloud API in background. Methods developed and used to this issues can be seen in `sinergym/sinergym/utils/gcloud.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/gcloud.py>`__ or in :ref:`API reference`.

*******************************************
Receiving experiments in remote containers
*******************************************

This script, called *DRL_battery.py*, will be allocated in every remote container and it is used to understand experiments command exposed above:

.. literalinclude:: ../../../DRL_battery.py
    :language: python

The list of parameter is pretty large. Let's see it:

- ``--environment`` or ``-env``: Environment name you want to use (see :ref:`Environments`)
- ``--episodes`` or ``-ep``: Number of episodes you want to train agent in simulation (Depending on environment episode length can be different)
- ``--algorithm`` or ``-alg``: Algorithm you want to use to train (Currently, it is available *PPO*, *A2C*, *DQN*, *DDPG* and *SAC*)
- ``--reward`` or ``-rw``: Reward class you want to use for reward function. Currently, possible values are "linear" and "exponential"(see :ref:`Rewards`).
- ``--normalization`` or ``-norm``: Apply normalization wrapper to observations during training. If it isn't specified wrapper will not be applied (see :ref:`Wrappers`).
- ``--multiobs`` or ``-mobs``: Apply Multi-Observation wrapper to observations during training. If it isn't specified wrapper will not be applied (see :ref:`Wrappers`).
- ``--logger`` or ``-log``: Apply Sinergym logger wrapper during training. If it isn't specified wrapper will not be applied (see :ref:`Wrappers` and :ref:`Logger`).
- ``--tensorboard`` or ``-tens``: This parameter will contain a path-file to allocate tensorboard training logs. If it isn't specified this log will be deactivate (see :ref:`DRL Logger`).
- ``--evaluation`` or ``-eval``: If it is specified, evaluation callback will be activate, else model evaluation will be deactivate during training (see :ref:`Deep Reinforcement Learning Integration`).
- ``--eval_freq`` or ``-evalf``: Only if ``--evaluation`` flag has been written. Episode frequency for evaluation.
- ``--eval_length`` or ``-evall``: Only if ``--evaluation`` flag has been written. Number of episodes for each evaluation.
- ``--log_interval`` or ``-inter``: This parameter is used for ``learn()`` method in each algorithm. It is important specify a correct value.
- ``--seed`` or ``-sd``: Seed for training, random components in process will be able to be recreated.
  
- **algorithm hyperparameters**: Execute ``python DRL_battery --help`` for more information.

********************
Google Cloud Alerts
********************

**Google Cloud Platform** include functionality in order to trigger some events and generate alerts in consequence. 
Then, a trigger has been created in our gcloud project which aim to advertise when an experiment has finished.
This alert can be captured in several ways (slack, sms, email, etc).
If you want to do the same, please, check Google Cloud Alerts documentation `here <https://cloud.google.com/monitoring/alerts>`__.

