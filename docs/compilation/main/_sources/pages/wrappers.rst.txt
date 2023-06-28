############
Wrappers
############

*Sinergym* has several **wrappers** in order to add some functionality in the environment 
that it doesn't have by default. The code can be found in 
`sinergym/sinergym/utils/wrappers.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/wrappers.py>`__.
You can implement your own wrappers inheriting from *gym.Wrapper* or some of its variants.

- **MultiObjectiveReward**: Environment step will return a vector reward (selected elements in wrapper constructor, 
  one for each objective) instead of a traditional scalar value. See `#301 <https://github.com/ugr-sail/sinergym/issues/301>`__.

- **PreviousObservationWrapper**: Wrapper to add observation values from previous timestep to current environment observation.
  It is possible to select the variables you want to track its previous observation values.

- **DatetimeWrapper**: Wrapper to substitute day value by is_weekend flag, and hour and month by sin and cos values. 
  Observation space is updated automatically.

- **DiscreteIncrementalWrapper**: A wrapper for an incremental setpoint action space environment. This wrapper
  will update a *continuous* environment, converting it in a *discrete* environment with an action mapping and action space 
  depending on the step and delta specified. The action will be sum with current setpoint values instead of overwrite the latest action. 
  Then, the action is the current setpoints values with the increase instead of the index of the action_mapping for the continuous 
  environment in the next layer. 

.. image:: /_static/incremental_wrapper.png
  :scale: 50 %
  :alt: Incremental wrapper graph.
  :align: center

- **NormalizeObservation**: It is used to transform observation received from simulator in values between 0 and 1.
  It is possible to define a list of variables you want to normalize, if you don't define this list, all environment
  variables will be included. In order to normalize values, it is necessary a dictionary in order to store max and min values. 
  Then, if a environment variable is not included in the dictionary specified in wrapper constructor, then the normalization 
  for that variable will be skipped.

- **LoggerWrapper**: Wrapper for logging all interactions between agent and environment. Logger class can be selected
  in the constructor if other type of logging is required. For more information about *Sinergym* Logger visit :ref:`Logger`.

- **MultiObsWrapper**: Stack observation received in a history queue (size can be customized).

.. note:: For examples about how to use these wrappers, visit :ref:`Wrappers example`.

.. warning:: The **order of wrappers** if you are going to use several at the same time is really important.
             The correct order is the same than the list shown above or subsets of that order. 

.. warning:: If you add new observation variables to the environment than the default ones, you have 
             to update the **value range dictionary** in `sinergym/sinergym/utils/constants.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/constants.py>`__ 
             so that normalization can be applied correctly. Otherwise, the variable normalization will be skipped.