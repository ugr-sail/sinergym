###########
Controllers
###########

*Sinergym* offers the possibility to implement custom controllers. These controllers can be employed as a baseline for comparing with more sophisticated control strategies, such as Reinforcement Learning algorithms.

Currently, we provide a **random controller** and a **rule-based controller** for both the *Datacenter* building and the *5Zone* building. See `controllers.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/controllers.py>`__ for implementation details.

The operation of these controllers is simple: given an observation from the environment, the controllers return an action that is applied to the environment. The executed action may depend on a set of user-defined rules that utilize the observed values (e.g., internal temperature, occupancy, etc.).

For an example of usage, refer to Section :ref:`Rule Controller example`.

.. warning:: Make sure that the variables used by your controller are part of the observation space of the environment.