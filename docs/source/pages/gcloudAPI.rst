#########################
Sinergym Google Cloud API 
#########################

In this project, an API based on RESTfull API for gcloud has been designed and developed in order to use Google Cloud infrastructure directly writing experiments definition ir our personal computer.

.. image:: /_static/Sinergym_cloud_API.png
  :width: 1000
  :alt: Sinergym cloud API diagram
  :align: center

From our personal computer, we send a list of experiments we want to be executed in Google Cloud, using **cloud_manager.py** script for that purpose. An instance will be created for every experiment defined.
Each VM send MLFlow logs to **MLFlow tracking server** (see :ref:`Mlflow tracking server set up` for details). On the other hand, Sinergym output and Tensorboard output are sent to a **Google Cloud Bucket** (see :ref:`Remote Tensorboard log` for details)

When an instance has finished its job, container **auto-remove** its host instance from Google Cloud Platform. Whether an instance is the last in the MIG, that container auto-remove the empty MIG too.

.. warning:: Don't try to remove an instance inner MIG directly using Google Cloud API REST, it needs to be executed from MIG to work. Some other problems (like wrong API REST documentation) have been solved in our API. We recommend you use this API directly.

Let’s see a detailed explanation above.

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
        --group_name sinergym-group \
        --experiment_commands 
        'python3 DRL_battery.py --environment Eplus-5Zone-hot-discrete-v1 --episodes 2 --algorithm DQN --logger --log_interval 1 --seed 58 --evaluation --eval_freq 1 --eval_length 1 --tensorboard ./tensorboard_log --remote_store' \
        'python3 DRL_battery.py --environment Eplus-5Zone-hot-continuous-v1 --episodes 3 --algorithm PPO --logger --log_interval 300 --seed 52 --evaluation --eval_freq 1 --eval_length 1 --tensorboard ./tensorboard_log --remote_store'

This example generates only 2 machines inner an instance group in your Google Cloud Platform because of you have defined two experiments. If you defined more experiments, more machines will be created by API.

This script do the next:

    1. Counting commands list in ``--experiment_commands`` parameter and generate an Managed Instance Group (MIG) with the same size.
    2. Waiting for **process 1** finishes.
    3. If *experiments-storage* Bucket doesn't exist, this script create one to store experiment result, else use the current one.
    4. Looking for instance names generated randomly by Google cloud once MIG is created (waiting for instances generation if they haven't been created yet).
    5. To each commands experiment, it is added ``--group_name`` option in order to each container see what is its own MIG (useful to auto-remove them).
    6. Looking for *id container* about each instance. This process waits for containers are initialize, since instance is initialize earlier than inner container.
    7. Sending each experiment command in containers from each instance using an SSH connection. 

.. note:: Because of its real-time process. Some containers, instance list action and others could take time. In that case, the API wait a process finish to execute the next (when it is necessary).

.. note:: This script uses gcloud API in background. Methods developed and used to this issues can be seen in `sinergym/sinergym/utils/gcloud.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/gcloud.py>`__ or in :ref:`API reference`.
    Remember to configure Google Cloud account correctly before use this functionality.

*******************************************
Receiving experiments in remote containers
*******************************************

This script, called *DRL_battery.py*, will be allocated in every remote container and it is used to understand experiments command exposed above by *cloud_manager.py* (``--experiment_commands``):

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
- ``--remote_store`` or ``-sto``: Determine if sinergym output will be sent to a common resource (Bucket), else will be allocate in remote container memory.
- ``--group_name`` or ``-group``: Added by *cloud_manager.py* automatically. It specify to which MIG the host instance belongs.
  
- **algorithm hyperparameters**: Execute ``python DRL_battery --help`` for more information.

This script do the next:

    1. Setting an appropriate name for the experiment. Following the next format: ``<algorithm>-<environment_name>-episodes<episodes_int>-seed<seed_value>(<experiment_date>)``
    2. Starting Mlflow track experiment with that name.
    3. Log all MlFlow parameters (including *sinergym.__version__*).
    4. Setting reward function specified in ``--reward`` parameter.
    5. Setting wrappers specified in environment.
    6. Defining model algorithm using hyperparameters.
    7. Calculate training timesteps using number of episodes.
    8. Setting up evaluation callback if it has been specified.
    9. Setting up Tensorboard logger callback if it has been specified.
    10. Training with environment.
    11. If ``--remote_store`` has been specified, saving all outputs in Google Cloud Bucket.
    12. Auto-delete remote container in Google Cloud Platform.

Containers permission to bucket storage output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you see in sinergym template explained in :ref:`4. Create your VM or MIG`, it is specified ``--scope``, ``--service-account`` and ``--container-env``. This aim to *remote_store* option in *DRL_battery.py* works correctly.
Those parameters provide each container with permissions to write in the bucket and manage Google Cloud Platform (auto instance remove function).
Container environment variables indicate zone, project_id and mlflow tracking server uri need it in :ref:`Mlflow tracking server set up`.

***********************
Remote Tensorboard log
***********************

In ``--tensorboard`` parameter we have to specify a **local path** in order to read this logs data in real-time. 
However, we want to send this data to *experiments-storage* bucket created in Google Cloud Platform by *cloud_manager.py* process (see :ref:`Executing API`).
Hence, Tensorboard has added support to do this recently (`#628 <https://github.com/ContinualAI/avalanche/pull/628>`__).
You have to write `gs://<bucket_name>/<tensorboard_log_path>` in *DRL_battery.py* ``--tensorboard`` parameter directly.

.. warning:: In the case that gs URI isn't recognized. Maybe is due to your tensorboard installation hasn't got access your google account. Try `gcloud auth application-default login <https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login>`__ command.

Visualize remote Tensorboard log in real-time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have two options:
    1. Create a remote server with tensorboard service deployed.
    2. Init that service in your local computer, reading from the bucket log, and access to the visualization in *http://localhost:6006*

The second options is enough since we can read from bucket when we need directly and shut down local service when we finish.

.. code:: sh

    $ tensorboard --logdir gs://experiments-storage/tensorboard_log/


******************************
Mlflow tracking server set up
******************************

Mlflow tracking server can be set up into your google account in order to organize your own experiments (:ref:`Mlflow`). You can separate **back-end** (SQL database) from tracking server.
In this way, you can shut down or delete server instance without loose your experiments run data, since SQL is always up. Let's see how:

.. literalinclude:: ../../../mlflowbuild.sh
    :language: sh

This bash script define all the process to configure this functionality automatically. (Once you execute it you don't have to use this script anymore). The arguments it needs are:
*PROJECT_ID*, *BUCKET_NAME*, *ZONE* and *DB_ROOT_PASSWORD*.

This script do the next for you:

    1. Creating Service account for mlflow service [mlflow-tracking-sa].
    2. Creating Back-end artifact bucket.
    3. Creating sql instance with root password specified in argument 4.
    4. Creating mlflow database inner SQL instance.
    5. Creating service account privileges to use Back-end [roles/cloudsql.editor]
    6. Generating an automatic script called **start_mlflow_tracking.sh** and sending to `gs://<BUCKET_NAME>/scripts/`
    7. Deleting local **start_mlflow_tracking.sh** file.
    8. Creating static external ip for mlflow-tracking-server
    9. Deploying remote server [mlflow-tracking-server]

Step 8 is very important, this allows you to delete server instance and create again when you need it without redefining server ip in sinergym-template for remote container experiments.
Notice that server instance creation use service account for mlflow, with this configuration mlflow can read from SQL server. In :ref:`4. Create your VM or MIG` it is specified MLFLOW_TRACKING_URI container environment variable using that external static ip.

.. warning:: It is important execute this script before create sinergym-template instances in order to anotate mlflow-server-ip.

.. note:: If you want to change any backend configuration, you can change any parameter of the script bellow.

********************
Google Cloud Alerts
********************

**Google Cloud Platform** include functionality in order to trigger some events and generate alerts in consequence. 
Then, a trigger has been created in our gcloud project which aim to advertise when an experiment has finished.
This alert can be captured in several ways (slack, sms, email, etc).
If you want to do the same, please, check Google Cloud Alerts documentation `here <https://cloud.google.com/monitoring/alerts>`__.

