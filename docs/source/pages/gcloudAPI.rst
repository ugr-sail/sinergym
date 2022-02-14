###########################
Sinergym with Google Cloud
###########################

In this project, we have defined some functionality based in gcloud API python in `sinergym/utils/gcloud.py`. Our time aim to configure a Google Cloud account and combine with Sinergym easily.

The main idea is to construct a **virtual machine** (VM) using **Google Cloud Engine** (GCE) in order to execute our **Sinergym container** on it. At the same time, this remote container will update a Google Cloud Bucket with experiments results and mlflow tracking server with artifacts if we configure that experiment with those options.

When an instance has finished its job, container **auto-remove** its host instance from Google Cloud Platform if experiments has been configured with this option.

Let’s see a detailed explanation above.

***********************
Preparing Google Cloud
***********************

1. First steps (configuration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, it is necessary that you have a Google Cloud account set up and SDK configured (auth, invoicing, project ID, etc). If you don't have this, it is recommended to check `their documentation <https://cloud.google.com/sdk/docs/install>`__.
Secondly, It is important to have installed `Docker <https://www.docker.com/>`__ in order to be able to manage these containers in Google Cloud.

You can link **gcloud** with **docker** accounts using the next (see `authentication methods <https://cloud.google.com/container-registry/docs/advanced-authentication>`__):

.. code:: sh

    $ gcloud auth configure-docker

If you don't want to have several problems in the future with the image build and Google Cloud functionality in general, we recommend you to **allow permissions for google cloud build** at the beginning (see `this documentation <https://cloud.google.com/build/docs/securing-builds/configure-access-for-cloud-build-service-account>`__).

.. image:: /_static/service-account-permissions.png
  :width: 500
  :alt: Permissions required for cloud build.
  :align: center

On the other hand, we are going to enable **Google Cloud services** in *API library*. These are API's which we need currently:

    - Google Container Registry API.
    - Artifact Registry API
    - Cloud Run API
    - Compute Engine API
    - Cloud Logging API
    - Cloud Monitoring API
    - Cloud Functions API
    - Cloud Pub/Sub API
    - Cloud SQL Admin API
    - Cloud Firestore API
    - Cloud Datastore API
    - Service Usage API
    - Cloud storage
    - Gmail API

Hence, you will have to allow this services into your **Google account**.
You can do it using **gcloud client SDK**:

.. code:: sh

    $ gcloud services list
    $ gcloud services enable artifactregistry.googleapis.com \
                             cloudapis.googleapis.com \
                             cloudbuild.googleapis.com \
                             containerregistry.googleapis.com \
                             gmail.googleapis.com \
                             sql-component.googleapis.com \
                             sqladmin.googleapis.com \
                             storage-component.googleapis.com \
                             storage.googleapis.com \
                             cloudfunctions.googleapis.com \
                             pubsub.googleapis.com \
                             run.googleapis.com \
                             serviceusage.googleapis.com \
                             drive.googleapis.com \
                             appengine.googleapis.com

Or you can use **Google Cloud Platform Console**:

.. image:: /_static/service-account-APIs.png
  :width: 800
  :alt: API's required for cloud build.
  :align: center

If you have installed *Sinergym* and *Sinergym extras*. **Google Cloud SDK must be linked with other python modules** in order to some functionality works in the future (for example, tensorboard). Please, execute the next in your terminal:

.. code:: sh

    $ gcloud auth application-default login

2. Use our container in Google Cloud Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our Sinergym container is uploaded in **Container Registry** as a public one currently. You can use it **locally**:

.. code:: sh

    $ docker run -it gcr.io/sinergym/sinergym:latest

If you want to use it in a **GCE VM**, you can execute the next:

.. code:: sh

    $ gcloud compute instances create-with-container sinergym \
        --container-image gcr.io/sinergym/sinergym \
        --zone europe-west1-b \
        --container-privileged \
        --container-restart-policy never \
        --container-stdin \
        --container-tty \
        --boot-disk-size 20GB \
        --boot-disk-type pd-ssd \
        --machine-type n2-highcpu-8

We have available containers in Docker Hub too. Please, visit our `repository <https://hub.docker.com/repository/docker/alejandrocn7/sinergym>`__

.. note:: It is possible to change parameters in order to set up your own VM with your preferences (see `create-with-container <https://cloud.google.com/sdk/gcloud/reference/compute/instances/create-with-container>`__).

.. warning:: ``--boot-disk-size`` is really important, by default VM set 10GB and it isn't enough at all for Sinergym container.
              This derive in a silence error for Google Cloud Build (and you would need to check logs, which incident is not clear).

3. Use your own container
~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have this repository forked and you want to upload **your own container on Google Cloud** and to use it. You can use **cloudbuild.yaml** 
with our **Dockerfile** for this purpose:

.. literalinclude:: ../../../cloudbuild.yaml
    :language: yaml

This file does the next:

    1. Write in **cache** for quick updates (if a older container was uploaded already).
    2. **Build** image (using cache if it's available)
    3. **Push** image built to Container Registry
    4. Make container public inner Container Registry.

There is an option section at the end of the file. Do not confuse this part with the virtual machine configuration. 
Google Cloud uses a helper VM to build everything mentioned above. At the same time, we are using this YAML file in order to
upgrade our container because of *PROJECT_ID* environment variable is defined by Google Cloud SDK, so its value is your current
project in Google Cloud global configuration.

.. warning:: In the same way VM needs more memory, Google Cloud Build needs at least 10GB to work correctly. In other case it may fail.

.. warning:: If your local computer doesn't have enough free space it might report the same error (there isn't difference by Google cloud error manager),
             so be careful.

In order to execute **cloudbuild.yaml**, you have to do the next:

.. code:: sh

    $ gcloud builds submit \
        --config ./cloudbuild.yaml .

*--substitutions* can be used in order to configure build parameters if they are needed.

.. note:: "." in ``--config`` refers to **Dockerfile**, which is necessary to build container image (see `build-config <https://cloud.google.com/build/docs/build-config>`__).

.. note:: In **cloudbuild.yaml** there is a variable named *PROJECT_ID*. However, it is not defined in substitutions. This is because it's a predetermined
          variable by Google Cloud. When build begins *"$PROJECT_ID"* is set to current value in gcloud configuration (see `substitutions-variables <https://cloud.google.com/build/docs/configuring-builds/substitute-variable-values>`__).

4. Create your VM or MIG
~~~~~~~~~~~~~~~~~~~~~~~~

To create a **VM** that uses this container, here there is an example:

.. code:: sh

    $ gcloud compute instances create-with-container sinergym \
        --container-image gcr.io/sinergym/sinergym \
        --zone europe-west1-b \
        --container-privileged \
        --container-restart-policy never \
        --container-stdin \
        --container-tty \
        --boot-disk-size 20GB \
        --boot-disk-type pd-ssd \
        --machine-type n2-highcpu-8

.. note:: ``--container-restart-policy never`` it's really important for a correct functionality.

.. warning:: If you decide enter in VM after create it immediately, it is possible container hasn't been created yet. 
             You can think that is an error, Google cloud should notify this. If this issue happens, you should wait for a several minutes.

To create a **MIG**, you need to create a machine set up **template** firstly, for example:

.. code:: sh

    $ gcloud compute instance-templates create-with-container sinergym-template \
    --container-image gcr.io/sinergym/sinergym \
    --container-privileged \
    --service-account storage-account@sinergym.iam.gserviceaccount.com \
    --scopes https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.full_control \
    --container-env=gce_zone=europe-west1-b,gce_project_id=sinergym,MLFLOW_TRACKING_URI=http://$(gcloud compute addresses describe mlflow-ip --format='get(address)'):5000 \
    --container-restart-policy never \
    --container-stdin \
    --container-tty \
    --boot-disk-size 20GB \
    --boot-disk-type pd-ssd \
    --machine-type n2-highcpu-8

.. note:: ``--service-account``, ``--scopes`` and ``--container-env`` parameters will be explained in :ref:`Containers permission to bucket storage output`. Please, read that documentation before using these parameters, since they require a previous configuration.

Then, you can create a group-instances as large as you want:

.. code:: sh

    $ gcloud compute instance-groups managed create example-group \
        --base-instance-name sinergym-vm \
        --size 3 \
        --template sinergym-template

.. warning:: It is possible that quote doesn't let you have more than one VM at the same time. Hence, the rest of VM's probably will be *initializing* always but never ready. If it is your case, we recommend you check your quotes `here <https://console.cloud.google.com/iam-admin/quotas>`__

5. Initiate your VM
~~~~~~~~~~~~~~~~~~~~

Your virtual machine is ready! To connect you can use ssh (see `gcloud-ssh <https://cloud.google.com/sdk/gcloud/reference/compute/ssh>`__):

.. code:: sh

    $ gcloud compute ssh <machine-name>

Google Cloud use a **Container-Optimized OS** (see `documentation <https://cloud.google.com/container-optimized-os/docs>`__) in VM. This SO have docker pre-installed with sinergym container.

.. image:: /_static/container1.png
  :width: 800
  :alt: GCE VM containers list
  :align: center


To use this container in our machine you only have to do:

.. code:: sh

    $ docker attach <container-name-or-ID>

.. image:: /_static/container2.png
  :width: 800
  :alt: GCE VM container usage.
  :align: center

And now you can execute your own experiments in Google Cloud! For example, you can enter in remote container with *gcloud ssh* and execute *DRL_battery.py* for the experiment you want.

********************************************
Executing experiments in remote containers
********************************************

This script, called *DRL_battery.py*, will be allocated in every remote container and it is used to execute experiments and combine it with **Google Cloud Bucket**, **Mlflow Artifacts**, **auto-remove**, etc:

.. literalinclude:: ../../../DRL_battery.py
    :language: python

.. note:: **DRL_battery.py** is able to be used to local experiments into client computer. For example, ``--auto_delete`` parameter will have no effect in experiment. This experiments results could be sent to bucket and mlflow artifacts if it is specified. We will see it.

The list of parameter is pretty large. Let's see it:

- ``--environment`` or ``-env``: Environment name you want to use (see :ref:`Environments`)
- ``--episodes`` or ``-ep``: Number of episodes you want to train agent in simulation (Depending on environment episode length can be different)
- ``--algorithm`` or ``-alg``: Algorithm you want to use to train (Currently, it is available *PPO*, *A2C*, *DQN*, *DDPG* and *SAC*)
- ``--reward`` or ``-rw``: Reward class you want to use for reward function. Currently, possible values are "linear" and "exponential"(see :ref:`Rewards`).
- ``--normalization`` or ``-norm``: Apply normalization wrapper to observations during training. If it isn't specified wrapper will not be applied (see :ref:`Wrappers`).
- ``--multiobs`` or ``-mobs``: Apply Multi-Observation wrapper to observations during training. If it isn't specified wrapper will not be applied (see :ref:`Wrappers`).
- ``--logger`` or ``-log``: Apply Sinergym logger wrapper during training. If it isn't specified wrapper will not be applied (see :ref:`Wrappers` and :ref:`Logger`).
- ``--tensorboard`` or ``-tens``: This parameter will contain a **path-file** or **path-remote-bucket** to allocate tensorboard training logs. If it isn't specified this log will be deactivate (see :ref:`DRL Logger`).
- ``--evaluation`` or ``-eval``: If it is specified, evaluation callback will be activate, else model evaluation will be deactivate during training (see :ref:`Deep Reinforcement Learning Integration`).
- ``--eval_freq`` or ``-evalf``: Only if ``--evaluation`` flag has been written. Episode frequency for evaluation.
- ``--eval_length`` or ``-evall``: Only if ``--evaluation`` flag has been written. Number of episodes for each evaluation.
- ``--log_interval`` or ``-inter``: This parameter is used for ``learn()`` method in each algorithm. It is important specify a correct value.
- ``--seed`` or ``-sd``: Seed for training, random components in process will be able to be recreated.
- ``--remote_store`` or ``-sto``: Determine if sinergym output and tensorboard log (when a local path is specified and not a remote bucket path) will be sent to a common resource (Bucket), else will be allocate in remote container memory only.
- ``--mlflow_store`` or ``-mlflow``: Determine if sinergym output and tensorboard log (when a local path is specified and not a remote bucket path) will be sent to a Mlflow Artifact, else will be allocate in remote container memory only.
- ``--group_name`` or ``-group``: It specify to which MIG the host instance belongs, it is important if --auto-delete is activated.
- ``--auto_delete`` or ``-del``: Whether this parameter is specified, remote instance will be auto removed when its job has finished.
  
- **algorithm hyperparameters**: Execute ``python DRL_battery --help`` for more information.

.. warning:: For a correct auto_delete functionality, please, use MIG's instead of individual instances.

This script do the next:

    1. Setting an appropriate name for the experiment. Following the next format: ``<algorithm>-<environment_name>-episodes<episodes_int>-seed<seed_value>(<experiment_date>)``
    2. Starting Mlflow track experiment with that name, if mlflow server is not available, it will be used an local path (*./mlruns*) in remote container.
    3. Log all MlFlow parameters (including *sinergym.__version__*).
    4. Setting reward function specified in ``--reward`` parameter.
    5. Setting wrappers specified in environment.
    6. Defining model algorithm using hyperparameters.
    7. Calculate training timesteps using number of episodes.
    8. Setting up evaluation callback if it has been specified.
    9. Setting up Tensorboard logger callback if it has been specified.
    10. Training with environment.
    11. If ``--remote_store`` has been specified, saving all outputs in Google Cloud Bucket. If ``--mlflow_store`` has been specified, saving all outputs in Mlflow run artifact.
    12. Auto-delete remote container in Google Cloud Platform when parameter ``--auto_delete`` has been specified.

Containers permission to bucket storage output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you see in sinergym template explained in :ref:`4. Create your VM or MIG`, it is specified ``--scope``, ``--service-account`` and ``--container-env``. This aim to *remote_store* option in *DRL_battery.py* works correctly.
Those parameters provide each container with permissions to write in the bucket and manage Google Cloud Platform (auto instance remove function).
Container environment variables indicate zone, project_id and mlflow tracking server uri need it in :ref:`Mlflow tracking server set up`.

Hence, it is **necessary** to **set up this service account** and give privileges in order to that objective. Then, following `Google authentication documentation <https://cloud.google.com/docs/authentication/getting-started>`__ we will do the next:

.. code:: sh

    $ gcloud iam service-accounts create storage-account
    $ gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:storage-account@PROJECT_ID.iam.gserviceaccount.com" --role="roles/owner"
    $ gcloud iam service-accounts keys create PROJECT_PATH/google-storage.json --iam-account=storage-account@PROJECT_ID.iam.gserviceaccount.com
    $ export GOOGLE_CLOUD_CREDENTIALS= PROJECT_PATH/google-storage.json

In short, we create a new service account called **storage-account**. Then, we dote this account with *roles/owner* permission. The next step is create a file key (json) called **google-storage.json** in our project root (gitignore will ignore this file in remote).
Finally, we export this file in **GOOGLE_CLOUD_CREDENTIALS** in our local computer in order to gcloud SDK knows that it has to use that token to authenticate.

***********************
Remote Tensorboard log
***********************

In ``--tensorboard`` parameter we have to specify a **local path** or a **Bucket path**.

If we specify a **local path**, tensorboard logs will be stored in remote containers memory. If you have specified ``--remote_store`` or ``--mlflow_store``, this logs will be sent to those remote storage when experiment finishes.
One of the strengths of Tensorboard is the ability to see the data in real time as the training is running. Thus, it is recommended to define in ``--tensorboard`` the **bucket path** directly in order to send that information
as the training is generating it (see `this issue <https://github.com/ContinualAI/avalanche/pull/628>`__ for more information). In our project we have *gs://experiments-storage/tensorboard_log* but you can have whatever you want.

.. note:: If in ``--tensorboard`` you have specified a gs path, ``--remote_store`` or ``--mlflow_store`` parameters don't store tensorboard logs.

.. warning:: Whether you have written a bucket path, don't write ``/`` at the end (*gs://experiments-storage/tensorboard_log/*), this causes that real-time remote storage doesn't work correctly.

.. warning:: In the case that gs URI isn't recognized. Maybe is due to your tensorboard installation hasn't got access your google account. Try `gcloud auth application-default login <https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login>`__ command.

Visualize remote Tensorboard log in real-time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have two options:
    1. Create a remote server with tensorboard service deployed.
    2. Initiate that service in your local computer, reading from the bucket log, and access to the visualization in *http://localhost:6006*

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

This bash script defines all the process to configure this functionality automatically. (Once you execute it you don't have to use this script anymore). The arguments it needs are:
*PROJECT_ID*, *BUCKET_NAME*, *ZONE* and *DB_ROOT_PASSWORD*.

This script do the next for you:

    1. Creating Service account for mlflow service `[mlflow-tracking-sa]`.
    2. Creating Back-end artifact bucket.
    3. Creating SQL instance with root password specified in argument 4.
    4. Creating mlflow database inner SQL instance.
    5. Creating service account privileges to use Back-end `[roles/cloudsql.editor]`
    6. Generating an automatic script called **start_mlflow_tracking.sh** and sending to ``gs://<BUCKET_NAME>/scripts/``
    7. Deleting local **start_mlflow_tracking.sh** file.
    8. Creating static external IP for `mlflow-tracking-server`
    9. Deploying remote server `[mlflow-tracking-server]`

Step 8 is very important, this allows you to delete server instance and create again when you need it without redefining server IP in sinergym-template for remote container experiments.
Notice that server instance creation use service account for mlflow, with this configuration mlflow can read from SQL server. In :ref:`4. Create your VM or MIG` it is specified MLFLOW_TRACKING_URI container environment variable using that external static IP.

.. warning:: It is important execute this script before create sinergym-template instances in order to annotate `mlflow-server-ip`.

.. note:: If you want to change any backend configuration, you can change any parameter of the script bellow.

.. note:: Whether you have written ``--mlflow_store``, Sinergym outputs will be sent to mlflow server as artifacts. These artifacts will be stored in the same bucket where is allocated ``gs://<BUCKET_NAME>``

********************
Google Cloud Alerts
********************

**Google Cloud Platform** include functionality in order to trigger some events and generate alerts in consequence. 
Then, a trigger has been created in our gcloud project which aim to advertise when an experiment has finished.
This alert can be captured in several ways (Slack, SMS, Email, etc).
If you want to do the same, please, check Google Cloud Alerts documentation `here <https://cloud.google.com/monitoring/alerts>`__.

