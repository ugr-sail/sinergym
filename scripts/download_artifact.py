import wandb

entity = "ai-uponor"
project_name = "ai-smatrix-learning"
artifact_name = "SB3_ppo_2025-04-25_13-37_2861g4oc"
tags = ['v' + str(i) for i in range(1)]
# Specify the path to W&B run and download artifact
run = wandb.init(project=project_name)
for tag in tags:

    artifact_path = f"{entity}/{project_name}/{artifact_name}:{tag}"
    artifact = run.use_artifact(artifact_path)
    artifact_dir = artifact.download()
    print(artifact_dir)

run.finish()
