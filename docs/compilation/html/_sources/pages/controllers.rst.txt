############
Controllers
############

Sinergym has a section to implement your own controllers. Currently, we have developed a **random agent** and a **rule-based agent**.
You can find this code in `sinergym/sinergym/utils/controllers.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/controllers.py>`__.
it is very useful in order to perform benchmarks as a reference point to study DRL algorithms:

.. literalinclude:: ../../../sinergym/utils/controllers.py
    :language: python

The functionality is very simple; given an environment observation, these instances return an action to interact with the environment. You can develop your own
controllers or modify rules of ``RuleBasedController``, for example. An usage of these controllers could be the next:

.. literalinclude:: ../../../examples/rule_controller.py
    :language: python

