<div align="center">
  <img src="images/logo.png" width=50%><br><br>
</div>

</p>
  <p align="center">
    <a href="https://github.com/ugr-sail/sinergym/releases">
      <img alt="Github latest release" src="https://img.shields.io/github/release-date/ugr-sail/sinergym" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/commits/main">
      <img alt="Github last commit" src="https://img.shields.io/github/last-commit/ugr-sail/sinergym" />
    </a>
    <a href="https://pypi.org/project/sinergym/">
      <img alt="pypi version" src="https://img.shields.io/pypi/v/sinergym" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/stargazers">
      <img alt="pypi downloads" src="https://img.shields.io/pypi/dm/sinergym" />
    </a>
    <a href="https://codecov.io/gh/ugr-sail/sinergym">
      <img src="https://codecov.io/gh/ugr-sail/sinergym/branch/main/graph/badge.svg" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/graphs/contributors">
      <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/ugr-sail/sinergym" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/issues">
      <img alt="Github issues" src="https://img.shields.io/github/issues/ugr-sail/sinergym?color=0088ff" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/pulls">
      <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/ugr-sail/sinergym?color=0088ff" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/blob/main/LICENSE">
      <img alt="Github license" src="https://img.shields.io/github/license/ugr-sail/sinergym" />
    </a>
    <a href="https://www.python.org/downloads/release/python-3120/">
      <img alt="pypi Python version" src="https://img.shields.io/pypi/pyversions/sinergym" />
    </a>
    <a href="https://hub.docker.com/r/sailugr/sinergym/tags">
      <img alt="DockerHub last version" src="https://img.shields.io/docker/v/sailugr/sinergym?color=blue&label=Docker%20Image%20Version&logo=docker" />
    </a>
    <br />
    <br />
    <a href="https://code.visualstudio.com/">
      <img src="https://img.shields.io/badge/Supported%20by-VSCode%20Power%20User%20%E2%86%92-gray.svg?colorA=655BE1&colorB=4F44D6&style=for-the-badge"/>
    </a>
  </p>

<div align="center">
  <img src="images/general_blueprint.png" width=80%><br><br>
</div>

## About Sinergym

*Sinergym* provides a [Gymnasium](https://gymnasium.farama.org/)-based interface to interact with simulation engines such as *EnergyPlus*. This allows control in simulation time through custom controllers, including **reinforcement learning** agents.

For more information about *Sinergym*, refer to its [documentation](https://ugr-sail.github.io/sinergym/compilation/main/index.html).

## Main features

⚙️  **Simulation engines compatibility**. *Sinergym* is currently compatible with the [EnergyPlus Python API](https://energyplus.readthedocs.io/en/latest/api.html) for controller-building communication.

📊  **Benchmark environments**. Similar to *Atari* or *Mujoco*, *Sinergym* allows the use of benchmarking environments to test and compare RL algorithms or custom control strategies.

🛠️  **Custom experimentation**. *Sinergym* enables effortless customization of experimental settings. Users can create their own environments or customize pre-configured ones within *Sinergym*. Select your preferred reward functions, wrappers, controllers, and more!

🏠  **Automatic building model adaptation**. Automatic adaptation of building models to align with user-defined settings.

🪛  **Automatic actuator control**. Seamless management of building actuators via the Gymnasium interface. Users only need to specify actuator names, and *Sinergym* will do the rest.

🤖  **Stable Baselines 3 integration**. *Sinergym* is highly integrated with [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) algorithms, wrappers and callbacks.

✅  **Controller-agnostic**. Any controller compatible with the Gymnasium interface can be integrated with *Sinergym*.

☁️  **Google Cloud execution**. *Sinergym* provides several features to execute experiments in [Google Cloud](https://cloud.google.com/).

📈  **Weights & Biases logging**. Automate the logging of training and evaluation data, and record your models in the cloud. *Sinergym* facilitates reproducibility and cloud data storage through [Weights and Biases](https://wandb.ai/site) integration.

📒  **Notebook examples**. Learn how to get the most out of *Sinergym* through our [notebooks examples](https://github.com/ugr-sail/sinergym/tree/main/examples).

📚  **Extensive documentation, unit tests, and GitHub actions workflows**. *Sinergym* follows proper development practices facilitating community contributions.


<div align="center">
  <img src="images/operation_diagram.png"><br><br>
</div>

## Project structure

This repository is organized into the following directories:

- `sinergym/`: the source code of *Sinergym*.
- `docs/`: *Sinergym*'s documentation sources.
- `examples/`: notebooks with several examples illustrating how to use *Sinergym*.
- `tests/`: *Sinergym* tests code.
- `scripts/`: auxiliar and help scripts.

## Available environments

For a complete and up-to-date list of available environments, please refer to [our documentation](https://ugr-sail.github.io/sinergym/compilation/main/pages/environments.html#).

## Installation

Read [INSTALL.md](https://github.com/ugr-sail/sinergym/blob/main/INSTALL.md) for detailed installation instructions.

## Usage example

This is a simple script using *Sinergym*:

```python
import gymnasium as gym
import sinergym

# Create environment
env = gym.make('Eplus-datacenter-mixed-continuous-stochastic-v1')

# Initialization
obs, info = env.reset()
truncated = terminated = False

# Run episode
while not (terminated or truncated):
    action = env.action_space.sample() # random action selection
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

Several usage examples can be consulted [here](https://ugr-sail.github.io/sinergym/compilation/main/pages/notebooks/basic_example.html#Basic-example).

## Contributing

To report questions and issues, [open an issue](https://github.com/ugr-sail/sinergym/issues) following the provided templates. We appreciate your feedback!

Check out [CONTRIBUTING.md](https://github.com/ugr-sail/sinergym/blob/main/CONTRIBUTING.md) for specific details on how to contribute.

## Projects using Sinergym

The following are some of the projects using *Sinergym*:

- [Demosthen/ActiveRL](https://github.com/Demosthen/ActiveRL)
- [VectorInstitute/HV-Ai-C](https://github.com/VectorInstitute/HV-Ai-C)
- [rdnfn/beobench](https://github.com/rdnfn/beobench)

📝 If you want to appear in this list, feel free to open a pull request and include the following badge in your repository:

<p align="center">
  <a href="https://github.com/ugr-sail/sinergym">
      <img src="https://img.shields.io/badge/Powered%20by-Sinergym%20%E2%86%92-gray.svg?colorA=00BABF&colorB=4BF2F7&style=for-the-badge"/>
  </a>
</p>

## Repository activity

![Alt](https://repobeats.axiom.co/api/embed/d8dc96d423d6996351e728a2412dba2551f99cca.svg "Repobeats analytics image")

## Citing Sinergym

If you use *Sinergym* in your work, please cite our [paper](https://www.sciencedirect.com/science/article/abs/pii/S0378778824011915):

```bibtex
@article{Campoy2025sinergym,
  title = {Sinergym – A virtual testbed for building energy optimization with Reinforcement Learning},
  author = {Alejandro Campoy-Nieves and Antonio Manjavacas and Javier Jiménez-Raboso and Miguel Molina-Solana and Juan Gómez-Romero},
  journal   = {Energy and Buildings},
  volume = {327},
  articleno = {115075},
  year = {2025},
  issn = {0378-7788},
  doi = {10.1016/j.enbuild.2024.115075},
  url = {https://www.sciencedirect.com/science/article/pii/S0378778824011915},
}
```
