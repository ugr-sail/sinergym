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

- Not included by default in Sinergym environments.

- This class can be replaced by any other data storage system, such as a remote database if desired.

|

The idea is to enable modular logging across different methods or platforms. In this section, we provide a brief description 
of the wrappers that use this logger storage module. For a detailed explanation, visit the :ref:`Logger Wrappers` section. An usage example is included in :ref:`Logging and storing data with logger wrappers`.

LoggerWrapper
-------------

- Uses the ``LoggerStorage`` class to store all information during the controller-environment interaction flow.

- The environment will include a new attribute called ``data_logger``: an instance of ``LoggerStorage`` containing all the relevant information.

- This wrapper also implements functionality to use the logger with custom metrics and episode summary metrics (i.e., it is customizable, as shown in :ref:`Logger Wrapper personalization/configuration`).

CSVLogger
---------

- Works with the **LoggerWrapper** ``data_logger`` instance. Calculates summary metrics implemented to parse 
  and save data in CSV files during simulations (see :ref:`Output format`).

WandB Logger
-------------

- Works with the **LoggerWrapper** ``data_logger`` instance. Calculates summary metrics in real-time, recording information 
  on the Weights and Biases platform in real-time.

*****************
WandBOutputFormat
*****************

- Adds compatibility between the Stable Baselines 3 training logging system and the Weights and Biases platform.

- It can be declared as ``WandBLogger`` to consolidate information in a single Weights and Biases panel automatically.
  See example :ref:`Training a model` for more information.