import wandb

entity = "alejandro-campoy"
project_name = "test-project"
artifact_name = "Eplus-PPO-training-example_2025-05-23_19-36_e8tealmy"
tags = ['v' + str(i) for i in range(1)]
# Specify the path to W&B run and download artifact
api = wandb.Api()
for tag in tags:

    artifact_path = f"{entity}/{project_name}/{artifact_name}:{tag}"
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    print(artifact_dir)
