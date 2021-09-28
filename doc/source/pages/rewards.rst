#######
Rewards
#######

Defining a reward function is one of the most important things in reinforcement learning. Consequently, our team has designed an structure which let you use our
reward class or defining a new one and integrate in available environments if you want:

.. literalinclude:: ../../../sinergym/utils/rewards.py
    :language: python

``LinearReward()`` class implements an evaluation which consists in taking into account **power consumption** and **temperature comfort**. This class is used
inner environment as an attribute.

``ExpReward()`` class is the same than ``LinearReward()`` class, but comfort penalty is exponential instead of lineal.

Reward is always negative. This means that perfect reward would be 0 (perfect power consumption and perfect temperature comfort), we apply penalties in both factors.
Notice there are two temperature comfort ranges in that class, those ranges are used rely on the specific date on the simulation. Moreover, notice there are
two weights in the reward function, this allows you to adjust how important each aspect is when making a general evaluation of the environment.

By default, all environments in gym register will use LinearReward() with default parameters. However, this configuration can be overwriting in ``gym.make()``, for example:

.. code:: python
    
    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=ExpReward(energy_weight=0.5))

.. note:: *Currently, it is only available these classes. However, more reward functions could be designed in the future!*

