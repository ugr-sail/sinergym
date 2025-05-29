###########################################
Serialization and Configuration Management
###########################################

In addition to initializing environments using predefined identifiers, Sinergym supports saving and restoring complete environment configurations through the `to_dict` and `from_dict` methods. This functionality allows users to export, modify, and reload environment setups with ease, thereby promoting reproducibility and simplifying the sharing of experimental configurations.

.. code-block:: python

    config = env.to_dict()
    # Save as YAML or reuse directly
    new_env = EplusEnv.from_dict(config)

Although this method allows configuration persistence, serializing a Sinergym environment is not a trivial task. The environment may contain nested components, such as custom wrappers or references to complex objects, which are not easily expressed using basic data types.

To address this, Sinergym includes a dedicated serialization module (`sinergym.utils.serialization`) that leverages **PyYAML** to enable robust serialization and deserialization of both the environment and its associated wrappers. This module automates the transformation of complex Python objects into a YAML-compatible format, facilitating the saving and restoration of complete environment states.

****************************************
Environment Configuration Serialization
****************************************

When a Sinergym environment is instantiated, its configuration is automatically serialized and saved as a YAML file (`env_config.pyyaml`) within the environment's designated output folder. This behavior ensures that every experiment includes a traceable and reproducible environment setup by default, even when the environment has been modified using user-specific configurations.

**************************************
Wrapper Serialization and Restoration
**************************************

Wrappers applied to an environment are not included in the default environment serialization and must be handled separately. Sinergym provides two utility functions for this purpose:

- `get_wrappers_info`: Extracts metadata from all applied wrappers and serializes it.
- `apply_wrappers_info`: Reconstructs and reapplies wrappers based on previously saved metadata.

****************************************
Example: Serialization and Restoration
****************************************

Wrapper configurations are not automatically serialized like the environment configuration. To save and restore wrapper configurations, you need to manually serialize the wrapper information and then apply it when loading the environment.

.. code-block:: python

    import gymnasium as gym
    from sinergym.utils.common import get_wrappers_info, apply_wrappers_info
    from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction
    from sinergym.envs.eplus_env import EplusEnv
    import yaml

    # Environment creation and wrapping
    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    env = NormalizeObservation(env)
    env = NormalizeAction(env)

    # Serialize wrapper information
    # By default, this will be saved in the environment's output directory
    wrappers_info = get_wrappers_info(env) # Apply wrappers to the environment

By default, if no save path is specified, the wrapper configuration will be stored as `wrappers_config.pyyaml` in the environment's output folder (accessible via `env.get_wrapper_attr('workspace_path')``).

To reload a complete environment, including its wrappers:

.. code-block:: python
    
    # Load environment parameters
    with open('<env_output_folder>/env_config.pyyaml', 'r') as f:
        env_params = yaml.load(f, Loader=yaml.FullLoader)

    # Modify environment parameters if needed...

    # Restore environment
    env = EplusEnv.from_dict(env_params)

    # Restore wrappers (from file or dictionary)
    env = apply_wrappers_info(env, '<env_output_folder>/wrappers_config.pyyaml')
    # Alternatively:
    env = apply_wrappers_info(env, wrappers_info)

This feature is used in Sinergym's scripts in :ref:`Usage`.

*************
Key Features
*************

- **Automatic environment serialization** on initialization (saved as `env_config.pyyaml``).
- **Wrapper metadata extraction and persistence** via `get_wrappers_info`.
- **Flexible restoration**: Wrapper configurations can be reapplied using either a YAML file path or a Python dictionary.
- **Default paths**: If no path is provided, serialization files are stored in the environment’s output folder.
- **Experiment reproducibility**: Facilitates the sharing and exact reproduction of complex experimental setups.