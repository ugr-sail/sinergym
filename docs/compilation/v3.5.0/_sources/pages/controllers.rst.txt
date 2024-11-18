############
Controllers
############

*Sinergym* includes a section for implementing your own **controllers**. At present, 
we've developed a **random agent** and a **rule-based agent** for *5Zone* and *Datacenter* 
buildings. You can locate this code in 
`sinergym/sinergym/utils/controllers.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/controllers.py>`__. 
This is particularly useful for performing benchmarks as a reference point for studying DRL algorithms.

The functionality is straightforward; given an environment observation, these instances return an action 
to interact with the environment. You can create your own controllers or modify existing rules, 
for instance. An example of usage can be found in the section :ref:`Rule Controller example`.

.. warning:: You have to make sure that the variables used particularly for your controller 
             are part of the observation space of the configured environment.
