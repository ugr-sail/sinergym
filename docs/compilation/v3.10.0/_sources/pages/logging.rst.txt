#######################
Logging system overview
#######################

*Sinergym* offers multiple loggers to record useful data generated during simulations. These are detailed below.

**************
TerminalLogger
**************

- Displays basic information to standard output during *Sinergym* executions.

- Messages are structured in layers (``ENVIRONMENT``, ``MODELING``, ``WRAPPER``, ``SIMULATION``, ``REWARD``) and levels 
  (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).

- The levels of each module can be updated in execution time using the ``sinergym.set_logger_level``. For more information
  about how to use it, see :ref:`Set terminal logger level`.

*************
LoggerStorage
*************

- Enables recording and managing interaction data.

- This class can be replaced by any other data storage system, such as a remote database if desired.

We provide a brief description of the wrappers that use ``LoggerStorage``. For a detailed explanation, visit the :ref:`Logger Wrappers` section. An usage example is included in :ref:`Logging and storing data with logger wrappers`.

Interaction data with LoggerWrapper
-----------------------------------

- Uses the ``LoggerStorage`` class to store all information during the controller-environment interaction flow.

- The environment will include a new attribute called ``data_logger``: an instance of ``LoggerStorage`` containing all the relevant information.

- This wrapper also implements functionality to use the logger with custom metrics and episode summary metrics (i.e., it is customizable, as shown in :ref:`LoggerWrapper customization`).

Summary metrics with CSVLogger
------------------------------

- The ``CSVLogger`` uses a ``data_logger`` instance. Calculates summary metrics implemented to parse 
  and save data in CSV files during simulations (see :ref:`Sinergym output`).

Remote logging using WandBLogger
--------------------------------

- The ``WandBLogger`` uses a ``data_logger`` instance. Calculates summary metrics in real-time, recording information 
  on the Weights and Biases platform in real-time.

*****************
WandBOutputFormat
*****************

- Adds compatibility between the Stable Baselines 3 training logging system and the Weights and Biases platform.

- It can be declared as ``WandBLogger`` to consolidate information in a single Weights and Biases panel automatically.
  See example :ref:`Training a model` for more information.