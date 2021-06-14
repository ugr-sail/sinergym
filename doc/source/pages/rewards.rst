#######
Rewards
#######

Defining a reward function is one of the most important things in reinforcement learning. Consequently, our team has designed an structure which let you use our
reward class or defining a new one and integrate in available environments if you want:

.. literalinclude:: ../../../energym/utils/rewards.py
    :language: python

``SimpleReward()`` class implements an evaluation which consists in taking into account **power consumption** and **temperature comfort**. This class is used
inner environment as an attribute.

Reward is always negative. This means that perfect reward would be 0 (perfect power consumption and perfect temperature comfort), we apply penalties in both factors.
Notice there are two temperature comfort ranges in that class, those ranges are used rely on the specific date on the simulation. Moreover, notice also there are
two weights in the reward function, this allows you to adjust how important each aspect is when making a general evaluation of the environment.

.. note:: *Currently, it is only available this class. However, more reward functions could be designed in the future!*

