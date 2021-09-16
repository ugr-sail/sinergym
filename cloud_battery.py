import argparse
import energym.utils.gcloud as gcloud

parser = argparse.ArgumentParser(
    description='Process for run experiments in Google Cloud')
parser.add_argument(
    '--project_id',
    '-id',
    help='Your Google Cloud project ID.')
parser.add_argument(
    '--zone',
    '-zo',
    default='europe-west1-b',
    help='service Engine zone to deploy to.')
parser.add_argument('--experiments', '-exps', default=envs_id, nargs='+')
# parser.add_argument(
#     '--name', default='demo-instance', help='New instance name.')

args = parser.parse_args()
