import argparse
import subprocess
import os
import time
from pprint import pprint

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials
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
    service.instances().list(project=project, zone=zone)
    result = service.instances().list(project=project, zone=zone).execute()
    instance_objects = result['items'] if 'items' in result else None
    instances = []
    if instance_objects:
        instances = [instance['name'] for instance in instance_objects]
    if base_instances_names:
        instances = [
            instance for instance in instances if base_instances_names in instance]
    return instances


# def create_instance(service, project, zone, name, bucket):
#     """Create an individual instance in Google Cloud.

#     Args:
#         service: gcloud API service built previously
#         project: Project id from Google Cloud Platform
#         zone: Google Cloud Zone where instance will be
#         name: Name of the new instance
#         bucket:

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


def delete_instance(service, project, zone, name):
    """Delete an instance inner Google Cloud.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instance is
        name: The name of instance you want to delete

    Returns:
        str JSON: State response from API

    """
    request = service.instances().delete(project=project, zone=zone, instance=name)
    response = request.execute()
    pprint(response)


def delete_instance_group(service, project, zone, group_name):
    """Delete a whole instance group inner Google Cloud.

    Args:
        service: gcloud API service built previously
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instance is
        group_name: The name of instance group you want to delete

    Returns:
        str JSON: State response from API

    """
    request = service.instanceGroupManagers().delete(
        project=project, zone=zone, instanceGroupManager=group_name)
    response = request.execute()
    pprint(response)


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
    pprint(response)


def execute_remote_command_instance(instance_name, experiment_command):
    """Execute a specified command in an instance previously created inner Google Cloud (Terminal is free after sending command).

    Args:
        instance_name: Name of instance where command will be executed.
        experiment_command: Command that will be executed

    """
    # Command to extract containerID inner VM
    cmd1 = ['gcloud', 'compute', 'ssh', instance_name,
            '--command', 'docker ps -q --filter name=klt']
    containerID_process = subprocess.Popen(
        cmd1,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    containerID_process.wait()
    result = containerID_process.stdout.read().decode().strip()
    err = containerID_process.stderr.read().decode().strip()

    # Exception management
    if err:
        print(err)
        raise RuntimeError
    if not result:
        raise RuntimeError(
            'It is not possible to find out containerID from machine specified. Please, check docker ps filter.')

    # Command to execute remote experiment in container
    cmd2 = ['gcloud', 'compute', 'ssh', instance_name, '--container',
            result, '--command', experiment_command]
    remoteCommand_process = subprocess.Popen(
        cmd2, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
