#############
Usage example
#############


Sinergym uses the standard OpenAI gym API. So basic loop should be
something like:

.. literalinclude:: ../../../try_env.py
    :language: python

At first glance may appear that sinergym is only imported but never used, but by importing Sinergym all its :ref:`Environments`
are defined to be used, like the one used in the example 'Eplus-5Zone-hot-continuous-v1'.

Notice that a folder will be created in the working directory after
creating the environment. It will contain the EnergyPlus outputs
produced during the simulation.