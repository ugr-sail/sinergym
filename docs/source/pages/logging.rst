########################
Logging System Overview
########################

*Sinergym* implements several loggers to control all information flow during execution.

****************
TerminalLogger
****************

* Prints information to the default output during processes.

* Messages are structured in layers (``ENVIRONMENT``, ``MODELING``, ``WRAPPER``, ``SIMULATION``, ``REWARD``) and levels 
  (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).

***************
LoggerStorage
***************

* Efficiently stores interaction information as class attributes.

* Not included by default in Sinergym environments; it handles only data logic, storage, and management functionality.

* This class can be replaced by any other data storage system, such as a remote database if desired.

To utilize **LoggerStorage** in environment interactions, apply the **LoggerWrapper** from *Sinergym*, by default uses this class to store data. 
The idea is to enable modular logging across different methods or platforms. In this section, we provide a brief description 
of the wrappers that use this logger storage module, but for a detailed explanation, visit the :ref:`Logger Wrappers` section and we have an usage
example in :ref:`Logging and storing data with logger wrappers`.

Logger Wrapper
---------------

* Uses the *LoggerStorage* class to store all information during the environment interaction flow.

* The environment will have a new attribute called ``data_logger``, an instance of *LoggerStorage* containing all the information.

* This wrapper also implements functionality to use the logger with custom metrics and 
  episode summary metrics (customizable, see :ref:`Logger Wrapper personalization/configuration`).

CSV Logger
-----------

* Works with the **LoggerWrapper** ``data_logger`` instance and calculated summary metrics implemented to parse 
  and save data in CSV files during simulations (see :ref:`Output format`).

WandB Logger
-------------

* Works with the **LoggerWrapper** ``data_logger`` instance and calculated summary metrics to dump all information 
  on the WandB platform in real-time.

******************
WandBOutputFormat
******************

* Integrates compatibility for the Stable Baselines 3 training logging system with Weights and Biases platform.

* Then, can be used with *Sinergym* *WandBLogger* to consolidate information in a single Weights and Biases panel automatically.
  See example :ref:`Training a model` for more information.