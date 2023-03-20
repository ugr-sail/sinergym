############
Wrappers
############

*Sinergym* has several **wrappers** in order to add some functionality in the environment 
that it doesn't have by default. Currently, we have developed a **normalization wrapper**, 
**multi-observation wrapper**, **multi-objective wrapper** and **Logger wrapper**. The code can be found in 
`sinergym/sinergym/utils/wrappers.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/wrappers.py>`__.
You can implement your own wrappers inheriting from *gym.Wrapper* or some of its variants.

- **NormalizeObservation**: It is used to transform observation received from simulator in values between 0 and 1.

- **LoggerWrapper**: Wrapper for logging all interactions between agent and environment. Logger class can be selected
  in the constructor if other type of logging is required. For more information about *Sinergym* Logger visit :ref:`Logger`.

- **MultiObjectiveReward**: Environment step will return a vector reward (selected elements in wrapper constructor, 
  one for each objective) instead of a traditional scalar value. See `#301 <https://github.com/ugr-sail/sinergym/issues/301>`__.

- **MultiObsWrapper**: Stack observation received in a history queue (size can be customized).

.. note:: For examples about how to use these wrappers, visit :ref:`Wrappers example`.

.. warning:: The order of wrappers if you are going to use several at the same time is really important.
             The correct order is **Normalization - Logger - Multi-Objectives - MultiObs** and subsets (for example, *Normalization* - *Multiobs* is valid).

.. warning:: If you add new observation variables to the environment than the default ones, you have 
             to update the **value range dictionary** in `sinergym/sinergym/utils/constants.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/constants.py>`__ 
             so that normalization can be applied correctly. Otherwise, you will encounter bug `#249 <https://github.com/ugr-sail/sinergym/issues/249>`__.