############
Wrappers
############

Energym has several wrappers in order to add some functionality in the environment that it doesn't have by default. Currently, we have developed a **normalization wrapper**, 
**multi-observation wrapper** and **Logger wrapper**. The code can be found in `energym/energym/utils/wrappers.py <https://github.com/jajimer/energym/blob/main/energym/utils/wrappers.py>`__.
You can implement your own wrappers inheriting from *gym.Wrapper* or some of its variants:

.. literalinclude:: ../../../energym/utils/wrappers.py
    :language: python

An usage of these wrappers could be the next:

.. literalinclude:: ../../../examples/try_wrappers.py
    :language: python

.. warning:: The order of wrappers if you are going to use several at the same time is really important.
             The correct order is **Normalization - Logger - MultiObs** and subsets (for example, *Normalization - Multiobs* is valid).

.. note:: For more information about Energym Logger, visit :ref:`Logger`