#############
Usage example
#############


Sinergym uses the standard OpenAI gym API. So basic loop should be
something like:

.. literalinclude:: ../../../try_env.py
    :language: python

Notice that a folder will be created in the working directory after
creating the environment. It will contain the EnergyPlus outputs
produced during the simulation.