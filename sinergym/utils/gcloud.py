"""Utilities to use Google Cloud Python API and Google Cloud API REST easier."""

import glob
import os
from pathlib import Path

import requests
from google.cloud import storage

####################### GCLOUD SERVICE OWNER #######################


def init_storage_client() -> storage.Client:
    """Initiate gcloud storage client to send petitions.

    Returns:
        storage.Client: Google Cloud storage client object to ask resources.

    """
    client = storage.Client()
    return client

####################### GCLOUD BUCKETS MANIPULATION #######################


def read_from_bucket(client, bucket_name, blob_prefix):
    """Read a file or a directory (recursively) from specified bucket to local file system.

    Args:
        client: Google Cloud storage client object to ask resources.
        bucket_name: Origin bucket name where reading.
        blob_prefix: Path where you want to read data inner the bucket (excluding gs://<bucket_name>).
    """
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_prefix)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = '/'.join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name)


def upload_to_bucket(
        client: storage.Client,
        src_path: str,
        dest_bucket_name: str,
        dest_path: str):
    """Upload a file or a directory (recursively) from local file system to specified bucket.

    Args:
        client (storage.Client): Google Cloud storage client object to ask resources.
        src_path (str): Path to the local file or directory you want to send
        dest_bucket_name (str): Destination bucket name
        dest_path (str): Path where you want to store data inner the bucket
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

######## OPERATION DESIGNED TO BE EXECUTED FROM REMOTE CONTAINER ########


def get_service_account_token() -> str:
    """Get token authorization if container has a valid service account.

    Returns:
        str: Authorization token for send petition to Google Cloud accounts (with its account service privileges).
    """
    url_token = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token'
    headers_token = {'Metadata-Flavor': 'Google'}
    token = requests.get(url_token, headers=headers_token).json()[
        'access_token']
    return token


def _get_instance_group_len(
        instance_group_name: str,
        token: str) -> int:
    """Get number of instances in a specific Managed Instance Groups (MIG).

    Args:
        instance_group_name (str): Instance group name you want to know number of instances.
        token (str): str to auth in Google Cloud Account service from container

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


def delete_instance_MIG_from_container(
        instance_group_name: str,
        token: str) -> requests.Response:
    """Delete the instance group inner Managed Instance Groups where container is executing. Whether this VM is alone in MIG, MIG will be removed too.

    Args:
        instance_group_name (str): Instance group name where container is executing.
        token (str): str to auth in Google Cloud Account service from container

    Returns:
        requests.Response: REST response
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
