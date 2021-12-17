"""Utilities to use Google Cloud Python API and Google Cloud API REST easier."""

import subprocess
import os
import time
from pprint import pprint
import glob
import requests

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials
from google.cloud import storage


####################### GCLOUD SERVICE OWNER #######################


def init_gcloud_service():
    """Init gcloud service to do operations.

    Returns:
        service: Google Cloud API service resource with owner credentials.

    """
    credentials = GoogleCredentials.get_application_default()
    service = googleapiclient.discovery.build(
        'compute', 'v1', credentials=credentials)
    return service


def init_storage_client():
    """Init gcloud storage client to send petitions.

    Returns:
        client: Google Cloud storage client object to ask resources.

    """
    client = storage.Client()
    return client


def list_instances(service, project, zone, base_instances_names=None):
    """List instances names created in Google Cloud currently.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instances are
        base_instances_names: By default is None, it filters instances if contains that sub str or not.

    Returns:
        list(str): Name of the instances availables in Google Cloud.

    """
    result = service.instances().list(project=project, zone=zone).execute()
    instance_objects = result['items'] if 'items' in result else None
    instances = []
    if instance_objects:
        instances = [instance['name'] for instance in instance_objects]
    if base_instances_names:
        instances = [
            instance for instance in instances if base_instances_names in instance]
    return instances


def delete_instance(service, project, zone, name):
    """Delete an instance inner Google Cloud.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instance is
        name: The name of instance you want to delete

    Returns:
        Dict: State response from API

    """
    request = service.instances().delete(project=project, zone=zone, instance=name)
    response = request.execute()
    return response


def delete_instance_group(service, project, zone, group_name):
    """Delete a whole instance group inner Google Cloud.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instance is
        group_name: The name of instance group you want to delete

    Returns:
        Dict: State response from API

    """
    request = service.instanceGroupManagers().delete(
        project=project, zone=zone, instanceGroupManager=group_name)
    response = request.execute()
    return response


def create_instance_group(
        service,
        project,
        zone,
        size,
        template_name,
        group_name):
    """Create an instance group (MIG) in Google Cloud.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instances are
        size: Number of instances desired inner MIG
        template_name: template name for machine type definition, previously defined into your Google Cloud account.
        group_name: Base name for every machine inner MIG, this name will be concatenated with a different random str for every machine

    Returns:
        str JSON: State response from API

    """

    body_request = {
        "versions": [
            {
                "instanceTemplate": "global/instanceTemplates/" + template_name
            }
        ],
        "name": group_name,
        "targetSize": size
    }
    request = service.instanceGroupManagers().insert(
        project=project, zone=zone, body=body_request)
    response = request.execute()
    return response


def list_instance_groups(service, project, zone):
    """List instances groups names created in Google Cloud currently.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instances are

    Returns:
        list(str): Name of the group instances availables in Google Cloud.

    """
    request = service.instanceGroupManagers().list(project=project, zone=zone)
    while request is not None:
        response = request.execute()
    for instance_group_manager in response['items']:
        pprint(instance_group_manager)


def get_container_id(instance_name, base='klt'):
    """Get container id inner an instance.

    Args:
        instance_name: Name of instance.
        base: The base name of container for filter (substring "klt" will be in all create-with-container gcloud operations).

    Returns:
        str: Container id inner instance.

    """
    cmd = ['gcloud', 'compute', 'ssh', instance_name,
           '--command', 'docker ps -q --filter name=' + base]
    containerID_process = subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    containerID_process.wait()
    result = containerID_process.stdout.read().decode().strip()
    err = containerID_process.stderr.read().decode().strip()

    if err:
        print(err)

    return result


def execute_remote_command_instance(
        container_id,
        instance_name,
        experiment_command):
    """Execute a specified command in an instance previously created inner Google Cloud (Terminal is free after sending command).

    Args:
        instance_name: Name of instance.
        container_id: Container id inner instance where command will be executed.
        experiment_command: Command that will be executed

    """

    cmd = ['gcloud', 'compute', 'ssh', instance_name, '--container',
           container_id, '--command', experiment_command]
    remoteCommand_process = subprocess.Popen(
        cmd, shell=False, stdout=None, stderr=None)


# Google Cloud doc function
def wait_for_operation(service, project, zone, operation, operation_type=''):
    """Sleep script execution until response status is DONE.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instances are
        operation: Response id
        operation_type: (Optional), operation type for prints feedback to user.

    Returns:
        Dict: Response state after status DONE.

    """
    print('Waiting for operation {} to finish...'.format(operation_type))
    while True:
        result = service.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print('{} operation DONE successfully.'.format(operation_type))
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)

    ####################### GCLOUD BUCKETS MANIPULATION #######################


def create_bucket(client, bucket_name='experiments-storage', location='EU'):
    """Create bucket in Google Cloud.

    Args:
        client: Google Cloud storage client object to ask resources.
        bucket_name: Name of the bucket
        location: Location for the bucket.

    Returns:
        bucket: Bucket object.

    """
    bucket = client.create_bucket(
        bucket_name,
        location=location)
    return bucket


def get_bucket(client, bucket_name):
    """Get bucket object into Google Account using client.

    Args:
        client: Google Cloud storage client object to ask resources.
        bucket_name: Name of the bucket

    Returns:
        bucket: Bucket object.

    """
    bucket = client.get_bucket(bucket_name)
    return bucket


def upload_to_bucket(client, src_path, dest_bucket_name, dest_path):
    """Upload a file or a directory (recursively) from local file system to specified bucket.

    Args:
        client: Google Cloud storage client object to ask resources.
        src_path: Path to the local file or directory you want to send
        dest_bucket_name: Destination bucket name
        dest_path: Path where you want to store data inner the bucket
    """
    bucket = client.get_bucket(dest_bucket_name)
    if os.path.isfile(src_path):
        blob = bucket.blob(os.path.join(dest_path, os.path.basename(src_path)))
        blob.upload_from_filename(src_path)
        return
    for item in glob.glob(src_path + '/*'):
        if os.path.isfile(item):
            blob = bucket.blob(os.path.join(dest_path, os.path.basename(item)))
            blob.upload_from_filename(item)
        else:
            upload_to_bucket(client,
                             item, dest_bucket_name, os.path.join(
                                 dest_path, os.path.basename(item)))


def read_from_bucket(client, src_path, ori_bucket_name, ori_path):
    """Read a file or a directory (recursively) from specified bucket to local file system.

    Args:
        client: Google Cloud storage client object to ask resources.
        src_path: Path to the local file or directory you want to read and download data
        ori_bucket_name: Origin bucket name where reading
        ori_path: Path where you want to read data inner the bucket
    """
    bucket = client.get_bucket(ori_bucket_name)
    blobs = bucket.list_blobs(prefix=ori_path)
    for blob in blobs:
        filename = blob.name.replace('/', '_')
        blob.download_to_filename(src_path + filename)


######## OPERATION DESIGNED TO BE EXECUTED FROM REMOTE CONTAINER ########


def get_service_account_token():
    """Get token authorization if container has a valid service account.

    Returns:
        str: Authorization token for send petition to Google Cloud accounts (with its account service privileges).
    """
    url_token = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token'
    headers_token = {'Metadata-Flavor': 'Google'}
    token = requests.get(url_token, headers=headers_token).json()[
        'access_token']
    return token


def _get_instance_group_len(instance_group_name, token):
    """Get number of instances in a specific Managed Instance Groups (MIG).

    Args:
        instance_group_name: Instance group name you want to know number of instances.
        token: str to auth in Google Cloud Account service from container

    Returns:
        int: Number of instances inner Managed Instance Groups
    """
    url_list = 'https://compute.googleapis.com/compute/v1/projects/' + \
        os.environ['gce_project_id'] + '/zones/' + os.environ['gce_zone'] + '/instanceGroupManagers/' + instance_group_name + '/listManagedInstances'
    header_auth = {'Authorization': 'Bearer ' + token}
    response = requests.post(
        url_list,
        headers=header_auth)

    return len(response.json()['managedInstances'])


def delete_instance_from_container(token):
    """Delete an individual instance group (this functionality doesn't work in Managed Instance Groups) where container is executing.

    Args:
        token: str to auth in Google Cloud Account service from container

    Returns:
        Request object: REST reponse
    """
    # Make request for delete host container
    url_delete = 'https://www.googleapis.com/compute/v1/projects/' + \
        os.environ['gce_project_id'] + '/zones/' + os.environ['gce_zone'] + '/instances/' + os.environ['HOSTNAME']
    header_auth = {'Authorization': 'Bearer ' + token}
    response = requests.delete(url_delete, headers=header_auth)
    return response


def delete_instance_MIG_from_container(instance_group_name, token):
    """Delete the instance group inner Managed Instance Groups where container is executing. Whether this vm is alone in MIG, MIG will be removed too.

    Args:
        instance_group_name: Instance group name where container is executing.
        token: str to auth in Google Cloud Account service from container

    Returns:
        Request object: REST reponse
    """
    header_auth = {'Authorization': 'Bearer ' + token}
    if _get_instance_group_len(instance_group_name, token) == 1:
        # We can delete entire instance group
        url_delete = 'https://compute.googleapis.com/compute/v1/projects/' + \
            os.environ['gce_project_id'] + '/zones/' + os.environ['gce_zone'] + '/instanceGroupManagers/' + instance_group_name
        response = requests.delete(url_delete, headers=header_auth)
    else:
        # We can only delete specific machine from instance group
        url_delete = 'https://compute.googleapis.com/compute/v1/projects/' + \
            os.environ['gce_project_id'] + '/zones/' + os.environ['gce_zone'] + '/instanceGroupManagers/' + instance_group_name + '/deleteInstances'

        data_delete = {
            "instances": [
                'zones/' +
                os.environ['gce_zone'] +
                '/instances/' +
                os.environ['HOSTNAME']],
            "skipInstancesOnValidationError": True}
        response = requests.post(
            url_delete,
            headers=header_auth,
            data=data_delete)
    return response
