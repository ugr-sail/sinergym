####################
Simple usage example
####################


Sinergym uses the standard OpenAI gym API. Lets see how to create a basic loop.

First we need to include sinergym and create an environment, in our case using 'Eplus-demo-v1'

.. literalinclude:: ../../../examples/basic_example.py
    :language: python
    :start-after: import numpy as np
    :end-before: for i in range(1):

At first glance may appear that sinergym is only imported but never used, but by importing Sinergym all its :ref:`Environments`
are defined to be used, in this case 'Eplus-demo-v1' with all the information contained in the idf file and the cnf file.

After this simple definition we are ready to loop the episodes. In summary the code we need is something like this:

.. literalinclude:: ../../../examples/basic_example.py
    :language: python
    :start-after: while not done:
    :end-before: if info['month'] != current_month

After all the executions we can check the result using a simple print like:

.. literalinclude:: ../../../examples/basic_example.py
    :language: python
    :start-after: print('Reward: ', sum(rewards), info)
    :end-before: env.close()
sum(rewards))
And as always dont forget to close the environment:

.. literalinclude:: ../../../examples/basic_example.py
    :language: python
    :start-after: print('Reward: ', sum(rewards), info)

Notice that a folder will be created in the working directory after
creating the environment. It will contain the EnergyPlus outputs
produced during the simulation.

****************
Full code
****************

.. literalinclude:: ../../../examples/basic_example.py
    :language: python

****************
Example output
****************

.. literalinclude:: ../../../examples/basic_example_output.txt
    :language: txt