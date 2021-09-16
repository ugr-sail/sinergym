import argparse
import subprocess
import os
import time
from pprint import pprint

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials
#from six.moves import input


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
        service: gcloud API service built previusly
        project: Project id from Google Cloud Platform
        zone: Google Cloud Zone where instance is
        name: The name of instance you want to delete

    Returns:
        str JSON: State response from API

    """
    return service.instances().delete(
        project=project,
        zone=zone,
        instance=name).execute()


def delete_instance_group(service, project, zone, group_name):
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
        service: gcloud API service built previusly
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


def run_command(cmd):
    """Run a command on a remote system."""
    command = subprocess.Popen(
        cmd, shell=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    result = command.stdout.read().decode()
    err = command.stderr.read().decode()
    return result.strip(), err.strip()
    # return result if result else ssh.stderr.readlines()


def main(project, zone):
    credentials = GoogleCredentials.get_application_default()
    service = googleapiclient.discovery.build(
        'compute', 'v1', credentials=credentials)
    size = 2
    template_name = "energym-template"
    group_name = "sinergym"
    # create_instance_group(
    #     service,
    #     project,
    #     zone,
    #     size,
    #     template_name,
    #     group_name)
    # instances = list_instances(service, project, zone, "sinergym")
    # print(instances)
    # print('Instances in project %s and zone %s:' % (project, zone))
    # for instance in instances:
    #     print(' - ' + instance['name'])
    #delete_instance_group(service, project, zone, 'sinergym')
    cmd1 = ['gcloud', 'compute', 'ssh', 'sinergym-7pzz',
            '--command', 'docker ps -q --filter name=klt']
    output1, err1 = run_command(cmd1)
    cmd2 = ['gcloud', 'compute', 'ssh', 'sinergym-7pzz', '--container',
            output1, '--command', 'energyplus --version']
    output2, err2 = run_command(cmd2)
    print(output1)
    print(cmd2)
    print(err1)
    print(output2)
    print(err2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID.')
    # parser.add_argument(
    #     'bucket_name', help='Your Google Cloud Storage bucket name.')
    parser.add_argument(
        '--zone',
        default='europe-west1-b',
        help='service Engine zone to deploy to.')
    # parser.add_argument(
    #     '--name', default='demo-instance', help='New instance name.')

    args = parser.parse_args()

    main(args.project_id, args.zone)
