############
Controllers
############

*Sinergym* has a section to implement your own **controllers**. Currently, we have developed 
a **random agent** and a **rule-based agent** to *5Zone* and *Datacenter* buildings.
You can find this code in 
`sinergym/sinergym/utils/controllers.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/controllers.py>`__.
it is very useful in order to perform benchmarks as a reference point to study DRL algorithms.

The functionality is very simple; given an environment observation, these instances return 
an action to interact with the environment. You can develop your own
controllers or modify current rules, for example. You can see an example of usage in 
section :ref:`Rule Controller example`.

.. warning:: You have to make sure that the variables used particularly for your controller 
             are part of the observation space of the configured environment.

