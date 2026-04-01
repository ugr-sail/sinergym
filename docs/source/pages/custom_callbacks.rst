################
Custom callbacks
################

*Sinergym* lets you register **custom EnergyPlus Python API runtime callbacks** so your own Python code runs at chosen points in the EnergyPlus timestep loop. This is useful for logging or custom simulation control logic.

By default, Sinergym uses callbacks to manage the control flow within the simulation layer. This mechanism allows users to attach any additional processes in parallel to any of the callback points provided by the EnergyPlus API, granting full flexibility to implement custom logic tailored to specific problems. A step-by-step Jupyter walkthrough is available in the :ref:`example notebook <example-notebook-custom-callbacks>` (also listed under *Examples* in the documentation sidebar).

Overview
========

- **When it takes effect:** After you call :py:meth:`~sinergym.envs.eplus_env.EplusEnv.register_callback`, your functions are queued until the next run. They are actually connected to EnergyPlus when the simulation **starts**, i.e. when you call :py:meth:`~sinergym.envs.eplus_env.EplusEnv.reset`.
- **Episodes:** Once registered, your callbacks stay in place for **later episodes** until you remove them or create a fresh environment. *Sinergym*’s own callbacks (observations, actions, context, warmup, progress) are unrelated; clearing yours does not touch them.
- **Removing callbacks:** :py:meth:`~sinergym.envs.eplus_env.EplusEnv.clear_callbacks` drops every user-registered callback you had added.

What to call on the environment
================================

Typical Gymnasium usage wraps the base environment, so use ``get_wrapper_attr`` to reach these (or call them directly on an unwrapped :class:`~sinergym.envs.eplus_env.EplusEnv`):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Member
     - Role
   * - ``register_callback(callback_name, callback_func[, component_program_name])``
     - Queue a callback for the given EnergyPlus callback point.
   * - ``clear_callbacks()``
     - Remove all user-registered callbacks from the internal registry.
   * - ``callbacks`` (property)
     - Read-only view: mapping from callback point name to a list of registered **function names** (for debugging and introspection).

Callback function signature
===========================

For most callback points, the function must accept the EnergyPlus **state** handle::

    def my_callback(state):
        ...

For ``callback_user_defined_component_model``, the signature is still ``callback_func(state)``; you must also pass ``component_program_name`` so EnergyPlus can associate the callback with the correct UserDefined component model in the IDF (see below).

Valid ``callback_name`` values
==============================

The string must match one of EnergyPlus’s runtime callback registration names (same names as in the EnergyPlus Python API):

- ``callback_begin_new_environment``
- ``callback_after_new_environment_warmup_complete``
- ``callback_begin_zone_timestep_before_init_heat_balance``
- ``callback_begin_zone_timestep_after_init_heat_balance``
- ``callback_begin_zone_timestep_before_set_current_weather``
- ``callback_begin_system_timestep_before_predictor``
- ``callback_end_zone_timestep_before_zone_reporting``
- ``callback_end_zone_timestep_after_zone_reporting``
- ``callback_end_system_timestep_before_hvac_reporting``
- ``callback_end_system_timestep_after_hvac_reporting``
- ``callback_inside_system_iteration_loop``
- ``callback_after_predictor_before_hvac_managers``
- ``callback_after_predictor_after_hvac_managers``
- ``callback_end_zone_sizing``
- ``callback_end_system_sizing``
- ``callback_unitary_system_sizing``
- ``callback_after_component_get_input``
- ``callback_user_defined_component_model``
- ``callback_register_external_hvac_manager``
- ``callback_message``
- ``callback_progress``

You can register **multiple** Python callables for the same ``callback_name`` by calling ``register_callback`` several times.

UserDefined component model (``callback_user_defined_component_model``)
=======================================================================

If ``callback_name`` is ``callback_user_defined_component_model``, you **must** set ``component_program_name`` to the program name of the UserDefined component model as declared in the IDF (for example ``'MyUserDefinedCoil'``). For every other callback name, ``component_program_name`` must be ``None``.

Reading building or simulation values inside a callback
=======================================================

Your callback receives EnergyPlus’s **state** handle. To read variables you have configured on the environment, get ``energyplus_simulator`` from the env and use ``exchange`` with ``var_handlers`` (or other handles you set up). The example below registers a callback and reads a zone temperature:

.. code-block:: python

    def my_custom_callback(state):
        simulator = env.get_wrapper_attr("energyplus_simulator")
        if simulator.var_handlers and "Zone_Temperature" in simulator.var_handlers:
            temp = simulator.exchange.get_variable_value(
                state,
                simulator.var_handlers["Zone_Temperature"],
            )
            print(f"Zone temperature: {temp}")

    env.get_wrapper_attr("register_callback")(
        "callback_begin_system_timestep_before_predictor",
        my_custom_callback,
    )

UserDefined registration:

.. code-block:: python

    def user_defined_callback(state):
        ...

    env.get_wrapper_attr("register_callback")(
        "callback_user_defined_component_model",
        user_defined_callback,
        component_program_name="MyUserDefinedCoil",
    )

Errors
======

Invalid ``callback_name``, missing ``component_program_name`` for UserDefined callbacks, or passing ``component_program_name`` for any other callback raise ``ValueError``. See the :ref:`API reference` for full signatures.

.. _example-notebook-custom-callbacks:

Example notebook
================

The notebook `Custom callback functions in Sinergym <notebooks/custom_callbacks.html>`__ walks through creating an environment, registering EnergyPlus runtime callbacks, and reading values from the simulator inside a callback. It lives in the repository as ``examples/custom_callbacks.ipynb`` and appears in the toctree under **Examples** alongside the other notebooks.
