###########################
Sinergym with Google Cloud
###########################

In this project, we have defined some functionality based in gcloud API 
python in ``sinergym/utils/gcloud.py``. Our team aim to configure a Google 
Cloud account and combine with *Sinergym* easily.

The main idea is to construct a **virtual machine** (VM) using 
**Google Cloud Engine** (GCE) in order to execute our **Sinergym container** 
on it. At the same time, this remote container will update **Weights and Biases tracking server** with artifacts 
if we configure that experiment with those options.

When an instance has finished its job, container **auto-remove** its host 
instance from Google Cloud Platform if experiments has been configured 
with this option.

***********************
Preparing Google Cloud
***********************

First steps (configuration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, it is necessary that you have a Google Cloud account set up and 
SDK configured (auth, invoicing, project ID, etc). If you don't have this, 
it is recommended to check `their documentation <https://cloud.google.com/sdk/docs/install>`__.
Secondly, It is important to have installed 
`Docker <https://www.docker.com/>`__ in order to be able to manage these 
containers in Google Cloud.

You can link **gcloud** with **docker** accounts using the next 
(see `authentication methods <https://cloud.google.com/container-registry/docs/advanced-authentication>`__):

.. code:: sh

    $ gcloud auth configure-docker

If you don't want to have several problems in the future with the image 
build and Google Cloud functionality in general, we recommend you to 
**allow permissions for google cloud build** at the beginning 
(see `this documentation <https://cloud.google.com/build/docs/securing-builds/configure-access-for-cloud-build-service-account>`__).

.. image:: /_static/service-account-permissions.png
  :width: 500
  :alt: Permissions required for cloud build.
  :align: center

On the other hand, we are going to enable **Google Cloud services** 
in *API library*. These are API's which we need currently:

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

If you have installed *Sinergym* and *Sinergym extras*. **Google Cloud SDK must 
be linked with other python modules** in order to some functionality works in 
the future. Please, execute the next in your terminal:

.. code:: sh

    $ gcloud auth application-default login

Use our container in Google Cloud Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our *Sinergym* container is uploaded in **Container Registry** as a public one 
currently. You can use it **locally**:

.. code:: sh

    $ docker run -it eu.gcr.io/sinergym/sinergym:latest

If you want to use it in a **GCE VM**, you can execute the next:

.. code:: sh

    $ gcloud compute instances create-with-container sinergym \
        --container-image eu.gcr.io/sinergym/sinergym \
        --zone europe-west1-b \
        --container-privileged \
        --container-restart-policy never \
        --container-stdin \
        --container-tty \
        --boot-disk-size 20GB \
        --boot-disk-type pd-ssd \
        --machine-type n2-highcpu-8

We have available containers in Docker Hub too. Please, visit 
our `repository <https://hub.docker.com/repository/docker/sailugr/sinergym>`__

.. note:: It is possible to change parameters in order to set 
          up your own VM with your preferences (see 
          `create-with-container <https://cloud.google.com/sdk/gcloud/reference/compute/instances/create-with-container>`__).

.. warning:: ``--boot-disk-size`` is really important, by default 
             VM set 10GB and it isn't enough at all for *Sinergym* container.
             This derive in a silence error for Google Cloud Build 
             (and you would need to check logs, which incident is not clear).

Use your own container
~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have this repository forked and you want to upload **your own 
container on Google Cloud** and to use it. You can use **cloudbuild.yaml** 
with our **Dockerfile** for this purpose:

.. literalinclude:: ../../../cloudbuild.yaml
    :language: yaml

This file does the next:

    1. Write in **cache** for quick updates 
       (if a older container was uploaded already).
    2. **Build** image (using cache if it's available)
    3. **Push** image built to Container Registry
    4. Make container public inner Container Registry.

There is an option section at the end of the file. Do not confuse 
this part with the virtual machine configuration. Google Cloud 
uses a helper VM to build everything mentioned above. At the same 
time, we are using this *YAML* file in order to upgrade our container 
because of ``PROJECT_ID`` environment variable is defined by Google 
Cloud SDK, so its value is your current project in Google Cloud 
global configuration.

.. warning:: In the same way VM needs more memory, Google Cloud 
             Build needs at least 10GB to work correctly. In other 
             case it may fail.

.. warning:: If your local computer doesn't have enough free space 
             it might report the same error (there isn't difference 
             by Google cloud error manager), so be careful.

In order to execute **cloudbuild.yaml**, you have to do the next:

.. code:: sh

    $ gcloud builds submit --region europe-west1 \
        --config ./cloudbuild.yaml .

``--substitutions`` can be used in order to configure build 
parameters if they are needed.

.. note:: "." in ``--config`` refers to **Dockerfile**, which is 
          necessary to build container image (see 
          `build-config <https://cloud.google.com/build/docs/build-config>`__).

.. note:: In **cloudbuild.yaml** there is a variable named *PROJECT_ID*. 
          However, it is not defined in substitutions. This is because 
          it's a predetermined variable by Google Cloud. When build begins 
          *"$PROJECT_ID"* is set to current value in gcloud configuration 
          (see `substitutions-variables <https://cloud.google.com/build/docs/configuring-builds/substitute-variable-values>`__).

Create your VM or MIG
~~~~~~~~~~~~~~~~~~~~~~~~

To create a **VM** that uses this container, here there is an example:

.. code:: sh

    $ gcloud compute instances create-with-container sinergym \
        --container-image eu.gcr.io/sinergym/sinergym \
        --zone europe-west1-b \
        --container-privileged \
        --container-restart-policy never \
        --container-stdin \
        --container-tty \
        --boot-disk-size 20GB \
        --boot-disk-type pd-ssd \
        --machine-type n2-highcpu-8

.. note:: ``--container-restart-policy never`` it's really important for a 
          correct functionality.

.. warning:: If you decide enter in VM after create it immediately, it is 
             possible container hasn't been created yet. 
             You can think that is an error, Google cloud should notify this. 
             If this issue happens, you should wait for a several minutes.

To create a **MIG**, you need to create a machine set up **template** 
firstly, for example:

.. code:: sh

    $ gcloud compute instance-templates create-with-container sinergym-template \
    --container-image eu.gcr.io/sinergym/sinergym \
    --container-privileged \
    --service-account storage-account@sinergym.iam.gserviceaccount.com \
    --scopes https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.full_control \
    --container-env=gce_zone=europe-west1-b,gce_project_id=sinergym \
    --container-restart-policy never \
    --container-stdin \
    --container-tty \
    --boot-disk-size 20GB \
    --boot-disk-type pd-ssd \
    --machine-type n2-highcpu-8

.. note:: ``--service-account``, ``--scopes`` and ``--container-env`` parameters 
          will be explained in :ref:`Containers permission to bucket storage 
          output`. Please, read that documentation before using these parameters, 
          since they require a previous configuration.

Then, you can create a group-instances as large as you want:

.. code:: sh

    $ gcloud compute instance-groups managed create example-group \
        --base-instance-name sinergym-vm \
        --size 3 \
        --template sinergym-template

.. warning:: It is possible that quote doesn't let you have more than one 
             VM at the same time. Hence, the rest of VM's probably will 
             be *initializing* always but never ready. If it is your case, 
             we recommend you check your quotes 
             `here <https://console.cloud.google.com/iam-admin/quotas>`__

Initiate your VM
~~~~~~~~~~~~~~~~~~~~

Your virtual machine is ready! To connect you can use ssh 
(see `gcloud-ssh <https://cloud.google.com/sdk/gcloud/reference/compute/ssh>`__):

.. code:: sh

    $ gcloud compute ssh <machine-name>

Google Cloud use a **Container-Optimized OS** (see 
`documentation <https://cloud.google.com/container-optimized-os/docs>`__) 
in VM. This SO have docker pre-installed with *Sinergym* container.

To use this container in our machine you only have to do:

.. code:: sh

    $ docker attach <container-name-or-ID>



And now you can execute your own experiments in Google Cloud! For example, 
you can enter in remote container with *gcloud ssh* and execute 
*train_agent.py* for the experiment you want.

********************************************
Executing experiments in remote containers
********************************************

`train_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/train_agent.py>`__ and 
`load_agent.py <https://github.com/ugr-sail/sinergym/blob/main/scripts/load_agent.py>`__
will be allocated in every remote container and it is used to execute experiments and evaluations,
being possible to combine with **Google Cloud Bucket**, **Weights and Biases**, **auto-remove**, etc:

.. note:: **train_agent.py** can be used in local experiments 
          and send output data and artifact to remote storage 
          such as wandb without configure cloud computing too. 

The structure of the JSON to configure the experiment or evaluation is specified in :ref:`How to use` section.

.. warning:: For a correct auto_delete functionality, please, use MIG's 
             instead of individual instances.

Containers permission to bucket storage output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you see in *Sinergym* template explained in :ref:`Create your VM or MIG`, 
it is specified ``--scope``, ``--service-account`` and ``--container-env``. 
This aim to *remote_store* option in *train_agent.py* works correctly.
Those parameters provide each container with permissions to write in the bucket 
and manage Google Cloud Platform (auto instance remove function).
Container environment variables indicate zone and project_id.

Hence, it is **necessary** to **set up this service account** and give privileges 
in order to that objective. Then, following 
`Google authentication documentation <https://cloud.google.com/docs/authentication/getting-started>`__ 
we will do the next:

.. code:: sh

    $ gcloud iam service-accounts create storage-account
    $ gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:storage-account@PROJECT_ID.iam.gserviceaccount.com" --role="roles/owner"
    $ gcloud iam service-accounts keys create PROJECT_PATH/google-storage.json --iam-account=storage-account@PROJECT_ID.iam.gserviceaccount.com
    $ export GOOGLE_CLOUD_CREDENTIALS= PROJECT_PATH/google-storage.json

In short, we create a new service account called **storage-account**. 
Then, we dote this account with *roles/owner* permission. The next step 
is create a file key (json) called **google-storage.json** in our project 
root (gitignore will ignore this file in remote).
Finally, we export this file in **GOOGLE_CLOUD_CREDENTIALS** in our local computer 
in order to gcloud SDK knows that it has to use that token to authenticate.

Visualize remote wandb log in real-time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You only have to enter in `Weights & Biases <https://wandb.ai/site>`__ and log in with your GitHub account.

********************
Google Cloud Alerts
********************

**Google Cloud Platform** include functionality in order to trigger some events and generate 
alerts in consequence. Then, a trigger has been created in our gcloud project which aim to 
advertise when an experiment has finished.
This alert can be captured in several ways (Slack, SMS, Email, etc).
If you want to do the same, please, check Google Cloud Alerts documentation 
`here <https://cloud.google.com/monitoring/alerts>`__.

