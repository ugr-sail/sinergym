import argparse
import os
import time
from pprint import pprint

import googleapiclient.discovery
#from six.moves import input


def list_instances(service, project, zone):
    peticion = service.instances().list(project=project, zone=zone)
    print(peticion.to_json())
    result = service.instances().list(project=project, zone=zone).execute()
    return result['items'] if 'items' in result else None


def create_instance(service, project, zone, name, bucket):
    # Get the latest Debian Jessie image.
    image_response = service.images().getFromFamily(
        project='debian-cloud', family='debian-9').execute()
    source_disk_image = image_response['selfLink']

    # Configure the machine
    machine_type = "zones/%s/machineTypes/n1-standard-1" % zone
    startup_script = open(
        os.path.join(
            os.path.dirname(__file__), 'startup-script.sh'), 'r').read()
    image_url = "http://storage.googleapis.com/gce-demo-input/photo.jpg"
    image_caption = "Ready for dessert?"

    config = {
        'name': name,
        'machineType': machine_type,

        # Specify the boot disk and the image to use as a source.
        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': source_disk_image,
                }
            }
        ],

        # Specify a network interface with NAT to access the public
        # internet.
        'networkInterfaces': [{
            'network': 'global/networks/default',
            'accessConfigs': [
                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
            ]
        }],

        # Allow the instance to access cloud storage and logging.
        'serviceAccounts': [{
            'email': 'default',
            'scopes': [
                'https://www.googleapis.com/auth/devstorage.read_write',
                'https://www.googleapis.com/auth/logging.write'
            ]
        }],

        # Metadata is readable from the instance and allows you to
        # pass configuration from deployment scripts to instances.
        'metadata': {
            'items': [{
                # Startup script is automatically executed by the
                # instance upon startup.
                'key': 'startup-script',
                'value': startup_script
            }, {
                'key': 'url',
                'value': image_url
            }, {
                'key': 'text',
                'value': image_caption
            }, {
                'key': 'bucket',
                'value': bucket
            }]
        }
    }

    return service.instances().insert(
        project=project,
        zone=zone,
        body=config).execute()


def delete_instance(service, project, zone, name):
    return service.instances().delete(
        project=project,
        zone=zone,
        instance=name).execute()


def create_instance_group(
        service,
        project,
        zone,
        size,
        template_name,
        group_name):

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


def main(project, zone):
    service = googleapiclient.discovery.build('compute', 'v1')
    print('Creating instance group.')
    size = 4
    template_name = "energym-template"
    group_name = "group-example"
    create_instance_group(
        service,
        project,
        zone,
        size,
        template_name,
        group_name)
    # instances = list_instances(service, project, zone)
    # print('Instances in project %s and zone %s:' % (project, zone))
    # for instance in instances:
    #     print(' - ' + instance['name'])


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
