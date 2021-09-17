import argparse
import energym.utils.gcloud as gcloud

parser = argparse.ArgumentParser(
    description='Process for run experiments in Google Cloud')
parser.add_argument(
    '--project_id',
    '-id',
    type=str,
    dest='project',
    help='Your Google Cloud project ID.')
parser.add_argument(
    '--zone',
    '-zo',
    type=str,
    default='europe-west1-b',
    dest='zone',
    help='service Engine zone to deploy to.')
parser.add_argument(
    '--experiment_commands',
    '-cmds',
    default=['python3 ./algorithm/DQN.py -env Eplus-demo-v1 -ep 1 -'],
    nargs='+',
    dest='commands',
    help='list of commands for DRL_battery.py you want to execute remotely.')
parser.add_argument(
    '--template_name',
    '-tem',
    type=str,
    default='energym-template',
    dest='template_name',
    help='Name of template previously created in gcloud account to generate VM copies.')
parser.add_argument(
    '--group_name',
    '-group',
    type=str,
    default='energym-group',
    dest='group_name',
    help='Name of instance group(MIG) will be created during experimentation.')

args = parser.parse_args()

n_experiments = len(args.commands)

service = gcloud.init_gcloud_service()

# Create instance group
gcloud.create_instance_group(
    service=service,
    project=args.project,
    zone=args.zone,
    size=n_experiments,
    template_name=args.template_name,
    group_name=args.group_name)

# Problem: VM is up earlier than container executing --> sleep execution?

# List VM names
instances = gcloud.list_instances(
    service=service,
    project=args.project,
    zone=args.zone,
    base_instances_names=args.group_name)
# Number of machines should be the same than commands
assert len(instances) == n_experiments

# Execute a comand in every container inner VM
for i, instance in enumerate(instances):
    gcloud.execute_remote_command_instance(
        instance_name=instance,
        experiment_command=args.commands[i])

# Close VM when finished with google cloud alerts?
