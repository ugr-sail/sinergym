#############
Usage Example
#############

Assuming you've used our Dockerfile for installation, the *try_env.py* 
file should already be in your workspace. If you've installed everything 
directly on your local machine, ensure this file is placed within our 
cloned repository. Regardless of your installation method, you should 
have a terminal ready with the appropriate Python version and *Sinergym* 
running correctly.

We'll begin with the most straightforward use case for the *Sinergym* tool. 
In the root repository, you'll find the script **try_env.py**:

.. literalinclude:: ../../../scripts/try_env.py
    :language: python

Upon initial inspection, it might seem that *Sinergym* is imported but not utilized. 
However, importing *Sinergym* defines all its 
`Environments <https://ugr-sail.github.io/sinergym/compilation/html/pages/environments.html>`__ 
for use. In this instance, ``Eplus-demo-v1`` is readily available with all its features.

We instantiate our environment using **gym.make** and execute the simulation for a single 
episode (``for i in range(1)``). The rewards returned by the environment are collected and their 
monthly average is computed.

Each step's action is randomly selected from its action space, as defined by the Gymnasium standard. 
Once the results are displayed and the episode concludes, we terminate the environment with `env.close()`.

.. important:: This represents the most basic usage example. Additional functional examples can be found 
               in the **Examples** section.