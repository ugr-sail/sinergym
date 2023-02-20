#############
Usage example
#############

If you used our Dockerfile during installation, you should have the *try_env.py* file 
in your workspace as soon as you enter in. In case you have installed everything on 
your local machine directly, place it inside our cloned repository.
In any case, we start from the point that you have at your disposal a terminal with 
the appropriate python version and Sinergym running correctly.

Let's start with the simplest use case for the Sinergym tool. In the root repository we have the script **try_env.py**:

.. literalinclude:: ../../../scripts/try_env.py
    :language: python

The **Sinergym import** is really important, because without it the ID's of our environments will not have been registered 
in the gymnasium module and therefore we cannot use our buildings as gymnasium environments.

We create our environment with **gym.make** and we run the simulation for one episode (`for i in range(1)`). 
We collect the rewards returned by the environment and calculate their average each month of simulation.

The action taken at each step is randomly chosen from its action space defined under the Gymnasium standard. 
When we have finished displaying the results on the screen and the episode is finished, we close the environment with `env.close()`.

.. note:: This is the simplest usage example. More functionality examples are shown in **Examples** section.