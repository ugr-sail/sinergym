import argparse
import subprocess
import os
import time
from pprint import pprint
import glob

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials
from google.cloud import storage
# from six.moves import input


def init_gcloud_service():
    """List instances names created in Google Cloud currently.

    Returns:
        service: Google Cloud API service resource with owner credentials.

    """
    credentials = GoogleCredentials.get_application_default()
    service = googleapiclient.discovery.build(
        'compute', 'v1', credentials=credentials)
    return service


def init_storage_client():
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

    # Exception management
    # if err:
    #     raise RuntimeError(err)
    # if not result:
    #     raise RuntimeError(
    #         'It is not possible to find out containerID from machine specified. Please, check docker ps filter.')
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
        cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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


def create_bucket(client, bucket_name, location):
    bucket = client.create_bucket(
        bucket_name,
        location=location)
    return bucket


def get_bucket(client, bucket_name):
    bucket = client.get_bucket(bucket_name)
    return bucket


def upload_to_bucket(client, src_path, dest_bucket_name, dest_path):
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


# def create_instance(service, project, zone, name, bucket):
#     """Create an individual instance in Google Cloud.

#     Args:
#         service: gcloud API service built previously
#         project: Project id from Google Cloud Platform
#         zone: Google Cloud Zone where instance will be
#         name: Name of the new instance
#         bucket: Bucket name used for create instance

#     Returns:
#         list(str): Name of the instances availables in Google Cloud.

#     """
#     # Get the latest Debian Jessie image.
#     image_response = service.images().getFromFamily(
#         project='debian-cloud', family='debian-9').execute()
#     source_disk_image = image_response['selfLink']

#     # Configure the machine
#     machine_type = "zones/%s/machineTypes/n1-standard-1" % zone
#     startup_script = open(
#         os.path.join(
#             os.path.dirname(__file__), 'startup-script.sh'), 'r').read()
#     image_url = "http://storage.googleapis.com/gce-demo-input/photo.jpg"
#     image_caption = "Ready for dessert?"

#     config = {
#         'name': name,
#         'machineType': machine_type,

#         # Specify the boot disk and the image to use as a source.
#         'disks': [
#             {
#                 'boot': True,
#                 'autoDelete': True,
#                 'initializeParams': {
#                     'sourceImage': source_disk_image,
#                 }
#             }
#         ],

#         # Specify a network interface with NAT to access the public
#         # internet.
#         'networkInterfaces': [{
#             'network': 'global/networks/default',
#             'accessConfigs': [
#                 {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
#             ]
#         }],

#         # Allow the instance to access cloud storage and logging.
#         'serviceAccounts': [{
#             'email': 'default',
#             'scopes': [
#                 'https://www.googleapis.com/auth/devstorage.read_write',
#                 'https://www.googleapis.com/auth/logging.write'
#             ]
#         }],

#         # Metadata is readable from the instance and allows you to
#         # pass configuration from deployment scripts to instances.
#         'metadata': {
#             'items': [{
#                 # Startup script is automatically executed by the
#                 # instance upon startup.
#                 'key': 'startup-script',
#                 'value': startup_script
#             }, {
#                 'key': 'url',
#                 'value': image_url
#             }, {
#                 'key': 'text',
#                 'value': image_caption
#             }, {
#                 'key': 'bucket',
#                 'value': bucket
#             }]
#         }
#     }

#     return service.instances().insert(
#         project=project,
#         zone=zone,
#         body=config).execute()
