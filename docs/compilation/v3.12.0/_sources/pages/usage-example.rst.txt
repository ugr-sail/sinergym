#############
Usage example
#############

Once *Sinergym* has been installed (either manually or via Docker), you can use the ``try_env.py`` script to test that everything is working correctly:

.. literalinclude:: ../../../scripts/try_env.py
    :language: python

The script instantiates a sample environment using ``gym.make``. It then runs a simulation. Note how the actions performed are randomly sampled from the action space.

Once the execution is complete, the resources are freed using ``env.close()``.

.. note:: After following this simple example, go to the examples section for more complex use cases.