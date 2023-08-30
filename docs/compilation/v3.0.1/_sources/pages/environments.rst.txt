############
Environments
############

As mentioned in introduction, *Sinergym* follows the next structure:

.. image:: /_static/sinergym_diagram.png
  :width: 800
  :alt: *Sinergym* backend
  :align: center

|

*Sinergym* is composed of three main components: *agent*,
*communication* interface and *simulation*. The agent sends actions and receives observations from the environment
through the Gymnasium interface. At the same time, the gym interface communicates with the simulator engine
via *EnergyPlus* Python API, which provide the functionality to manage handlers such as actuators, meters and variables,
so their current values have a direct influence on the course of the simulation. 

The next image shows this process more detailed:

.. image:: /_static/backend.png
  :width: 1600
  :alt: *Sinergym* backend
  :align: center

|

The *Modeling* module works at the same level as the API and allows to adapt the building models before the start of each 
episode. This allows that the API can work correctly with the user's definitions in the environment. 

This scheme is very abstract, since these components do some additional tasks such as handling the folder structure 
of the output, preparing the handlers before using them, initiating callbacks for data collection during simulation, 
and much more.

***********************************
Additional observation information
***********************************

In addition to the observations returned in the step and reset methods as you can see in the images above, 
both return a Python dictionary with additional information:

- **Reset info:** This dictionary has the next keys:

.. code-block:: python

  info = {
            'time_elapsed(hours)': # <Simulation time elapsed in hours>,
            'month': # <Month in which the episode starts.>,
            'day': # <Day in which the episode starts.>,
            'hour': # <Hour in which the episode starts.>,
            'is_raining': # <True if it is raining in the simulation.>,
            'timestep': # <Timesteps count.>,
        }

- **step info:** This dictionary has the same keys than reset info, but it is added the action sent (action sent to the
  simulation, not the action sent to the environment), the reward and reward terms. The reward terms depend on
  the reward function used.  

**************************
Environments List
**************************

The **list of available environments** is the following:

+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Env. name                                       | IDF file                              | EPW file                                                        | Weather variability | Action space | Simulation period |
+=================================================+=======================================+=================================================================+=====================+==============+===================+
| Eplus-demo-v1                                   | 5ZoneAutoDXVAV.idf                    | USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw           | No                  | Discrete(10) | 01/01 - 31/03     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-hot-discrete-v1                     | 5ZoneAutoDXVAV.idf                    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-mixed-discrete-v1                   | 5ZoneAutoDXVAV.idf                    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-cool-discrete-v1                    | 5ZoneAutoDXVAV.idf                    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-hot-continuous-v1                   | 5ZoneAutoDXVAV.idf                    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-mixed-continuous-v1                 | 5ZoneAutoDXVAV.idf                    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-cool-continuous-v1                  | 5ZoneAutoDXVAV.idf                    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-hot-discrete-stochastic-v1          | 5ZoneAutoDXVAV.idf                    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-mixed-discrete-stochastic-v1        | 5ZoneAutoDXVAV.idf                    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-cool-discrete-stochastic-v1         | 5ZoneAutoDXVAV.idf                    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-hot-continuous-stochastic-v1        | 5ZoneAutoDXVAV.idf                    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-mixed-continuous-stochastic-v1      | 5ZoneAutoDXVAV.idf                    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-5zone-cool-continuous-stochastic-v1       | 5ZoneAutoDXVAV.idf                    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-hot-discrete-v1                | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-hot-continuous-v1              | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-hot-discrete-stochastic-v1     | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-hot-continuous-stochastic-v1   | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-mixed-discrete-stochastic-v1   | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-mixed-continuous-v1            | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-mixed-discrete-v1              | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-mixed-continuous-stochastic-v1 | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-cool-discrete-stochastic-v1    | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-cool-continuous-v1             | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-cool-discrete-v1               | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-datacenter-cool-continuous-stochastic-v1  | 2ZoneDataCenterHVAC_wEconomizer.idf   | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-hot-discrete-v1                 | ASHRAE9012016_Warehouse_Denver.idf    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-hot-continuous-v1               | ASHRAE9012016_Warehouse_Denver.idf    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Box(5)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-hot-discrete-stochastic-v1      | ASHRAE9012016_Warehouse_Denver.idf    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-hot-continuous-stochastic-v1    | ASHRAE9012016_Warehouse_Denver.idf    | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Box(5)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-mixed-discrete-stochastic-v1    | ASHRAE9012016_Warehouse_Denver.idf    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-mixed-continuous-v1             | ASHRAE9012016_Warehouse_Denver.idf    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Box(5)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-mixed-discrete-v1               | ASHRAE9012016_Warehouse_Denver.idf    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-mixed-continuous-stochastic-v1  | ASHRAE9012016_Warehouse_Denver.idf    | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Box(5)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-cool-discrete-stochastic-v1     | ASHRAE9012016_Warehouse_Denver.idf    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-cool-continuous-v1              | ASHRAE9012016_Warehouse_Denver.idf    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Box(5)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-cool-discrete-v1                | ASHRAE9012016_Warehouse_Denver.idf    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-warehouse-cool-continuous-stochastic-v1   | ASHRAE9012016_Warehouse_Denver.idf    | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Box(5)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-hot-discrete-v1                    | ASHRAE9012016_OfficeMedium_Denver.idf | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-hot-continuous-v1                  | ASHRAE9012016_OfficeMedium_Denver.idf | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-hot-discrete-stochastic-v1         | ASHRAE9012016_OfficeMedium_Denver.idf | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-hot-continuous-stochastic-v1       | ASHRAE9012016_OfficeMedium_Denver.idf | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-mixed-discrete-stochastic-v1       | ASHRAE9012016_OfficeMedium_Denver.idf | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-mixed-continuous-v1                | ASHRAE9012016_OfficeMedium_Denver.idf | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-mixed-discrete-v1                  | ASHRAE9012016_OfficeMedium_Denver.idf | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-mixed-continuous-stochastic-v1     | ASHRAE9012016_OfficeMedium_Denver.idf | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-cool-discrete-stochastic-v1        | ASHRAE9012016_OfficeMedium_Denver.idf | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-cool-continuous-v1                 | ASHRAE9012016_OfficeMedium_Denver.idf | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-cool-discrete-v1                   | ASHRAE9012016_OfficeMedium_Denver.idf | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-office-cool-continuous-stochastic-v1      | ASHRAE9012016_OfficeMedium_Denver.idf | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-cool-continuous-v1             | OfficeGridStorageSmoothing.idf        | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-mixed-continuous-v1            | OfficeGridStorageSmoothing.idf        | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-hot-continuous-v1              | OfficeGridStorageSmoothing.idf        | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-cool-continuous-stochastic-v1  | OfficeGridStorageSmoothing.idf        | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-mixed-continuous-stochastic-v1 | OfficeGridStorageSmoothing.idf        | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-hot-continuous-stochastic-v1   | OfficeGridStorageSmoothing.idf        | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Box(4)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-cool-discrete-v1               | OfficeGridStorageSmoothing.idf        | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-mixed-discrete-v1              | OfficeGridStorageSmoothing.idf        | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-hot-discrete-v1                | OfficeGridStorageSmoothing.idf        | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-cool-discrete-stochastic-v1    | OfficeGridStorageSmoothing.idf        | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-mixed-discrete-stochastic-v1   | OfficeGridStorageSmoothing.idf        | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-officegrid-hot-discrete-stochastic-v1     | OfficeGridStorageSmoothing.idf        | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-cool-continuous-v1                   | ShopWithVandBattery.idf               | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-mixed-continuous-v1                  | ShopWithVandBattery.idf               | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-hot-continuous-v1                    | ShopWithVandBattery.idf               | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-cool-continuous-stochastic-v1        | ShopWithVandBattery.idf               | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-mixed-continuous-stochastic-v1       | ShopWithVandBattery.idf               | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-hot-continuous-stochastic-v1         | ShopWithVandBattery.idf               | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Box(2)       | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-cool-discrete-v1                     | ShopWithVandBattery.idf               | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-mixed-discrete-v1                    | ShopWithVandBattery.idf               | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-hot-discrete-v1                      | ShopWithVandBattery.idf               | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | No                  | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-cool-discrete-stochastic-v1          | ShopWithVandBattery.idf               | USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-mixed-discrete-stochastic-v1         | ShopWithVandBattery.idf               | USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw             | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+
| Eplus-shop-hot-discrete-stochastic-v1           | ShopWithVandBattery.idf               | USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw                        | Yes                 | Discrete(10) | 01/01 - 31/12     |
+-------------------------------------------------+---------------------------------------+-----------------------------------------------------------------+---------------------+--------------+-------------------+


.. note:: For more information about buildings (epJSON column) and weathers (EPW column),
          please, visit sections :ref:`Buildings` and :ref:`Weathers` respectively.

*********************
Available Parameters
*********************

With the **environment constructor** we can configure the complete **context** of our environment 
for experimentation, either starting from one predefined by *Sinergym* shown in the 
table above or creating a new one.

*Sinergym* initially provides **non-configured** buildings and weathers. Depending of these argument values, 
these files are updated in order to adapt it to this new features, this will be made by Sinergym automatically.
For example, using another weather file requires building location and design days update, using new observation 
variables requires to update the ``Output:Variable`` and ``Output:Meter`` fields, the same occurs with extra 
configuration context concerned with simulation directly, if weather variability is set, then a weather with noise 
will be used. These new building and weather file versions, is saved in the Sinergym output folder, leaving the original intact.

The next subsections will show which **parameters** are available and what their function are:

building file 
==============

The parameter ``building_file`` is the *epJSON* file, a new `adaptation <https://energyplus.readthedocs.io/en/latest/schema.html>`__ 
of *IDF* (Intermediate Data Format) where *EnergyPlus* building model is defined. These files are not configured for a particular
environment as we have mentioned. Sinergym does a previous building model preparation to the simulation, see the *Modeling* element
in *Sinergym* backend diagram.

Weather files
==============

The parameter ``weather_file`` is the *EPW* (*EnergyPlus* Weather) file name where **climate conditions** during 
a year is defined.

This parameter can be either a weather file name (``str``) as mentioned, or a list of different weather files (``List[str]``).
When a list of several files is defined, *Sinergym* will select an *EPW* file in each episode and re-adapt building 
model randomly. This is done in order to increase the complexity in the environment whether is desired. 

The weather file used in each episode is stored in *Sinergym* episode output folder, if **variability** 
(section :ref:`Weather Variability` is defined), the *EPW* stored will have that noise included.

Weather Variability
====================

**Weather variability** can be integrated into an environment using ``weather_variability`` parameter.

It implements the `Ornstein-Uhlenbeck process <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.4200&rep=rep1&type=pdf>`__
in order to introduce **noise** to the weather data episode to episode. Then, parameter established is a Python tuple of three variables
(*sigma*, *mu* and *tau*) whose values define the nature of that noise.

.. image:: /_static/ornstein_noise.png
  :scale: 80 %
  :alt: Ornstein-Uhlenbeck process noise with different hyperparameters.
  :align: center


Reward
=======

The parameter called ``reward`` is used to define the **reward class** (see section :ref:`Rewards`)
that the environment is going to use to calculate and return reward values each timestep.

Reward Kwargs
==============

Depending on the reward class that is specified to the environment, it may have **different arguments** 
depending on its type. In addition, if a user creates a new custom reward, it can have new parameters as well.

Moreover, depending on the building being used for the environment, the values of these reward parameters may 
need to be different, such as the comfort range or the energy and temperature variables of the simulation that 
will be used to calculate the reward.

Then, the parameter called ``reward_kwargs`` is a Python dictionary where we can **specify all reward class arguments** 
that they are needed. For more information about rewards, visit section :ref:`Rewards`.

Maximum Episode Data Stored in Sinergym Output
===============================================

*Sinergym* stores all the output of an experiment in a folder organized in sub-folders for each episode 
(see section :ref:`Output format` for more information). Depending on the value of the parameter ``max_ep_data_store_num``, 
the experiment will store the output data of the **last n episodes** set, where **n** is the value of the parameter.

In any case, if *Sinergym Logger* (See :ref:`Logger` section) is activate, ``progress.csv`` will be present with 
the summary data of each episode.

Time variables
===============

*EnergyPlus* Python API has several methods in order to extract information about simulation time in progress. The
argument ``time_variables`` is a list in which we can specify the name of the 
`API methods <https://energyplus.readthedocs.io/en/latest/datatransfer.html#datatransfer.DataExchange>`__ 
whose values we want to include in our observation.

By default, *Sinergym* environments will have the time variables ``month``, ``day_of_month`` and ``hour``.

Variables
==========

The argument called ``variables`` is a dictionary in which it is specified the ``Output:Variable``'s we want to include in
the environment observation. The format of each element, in order to *Sinergym* can process it, is the next:

.. code-block:: python

  variables = {
    # <custom_variable_name> : (<"Output:Variable" original name>,<variable_key>),
    # ...
  }

.. note:: For more information about the available variables in an environment, execute a default simulation with
          *EnergyPlus* engine and see RDD file generated in the output.

Meters
==========

In a similar way, the argument ``meters`` is a dictionary in which we can specify the ``Output:Meter``'s we want to include in
the environment observation. The format of each element must be the next:

.. code-block:: python

  meters = {
    # <custom_meter_name> : <"Output:Meter" original name>,
    # ...
  }

.. note:: For more information about the available meters in an environment, execute a default simulation with
          *EnergyPlus* engine and see MDD and MTD files generated in the output.

Actuators
==========

The argument called ``actuators`` is a dictionary in which we specify the actuators we want to control with gymnasium interface, the format
must be the next:

.. code-block:: python

  actuators = {
    # <custom_actuator_name> : (<actuator_type>,<actuator_value>,<actuator_original_name>),
    # ...
  }

.. important:: Actuators that have not been specified will be controlled by the building's default schedulers.

.. note:: For more information about the available actuators in an environment, execute a default control with
          *Sinergym* directly (empty action space) and see ``data_available.txt`` generated.

Action space
===========================

As you have been able to observe, by defining the previous arguments, a definition of the observation and action 
space of the environment is being made. ``time_variables``, ``variables`` and ``meters`` make up our environment 
*observation*, while the ``actuators`` alone make up the environment *action*:

.. image:: /_static/spaces_elements.png
  :scale: 35 %
  :alt: *EnergyPlus* API components that compose observation and action spaces in *Sinergym*.
  :align: center

This allows us to do a **dynamic definition** of spaces, *Sinergym* will adapt the building model.
Observation space is created automatically, but action space must be defined in order to set up
the range values supported by the Gymnasium interface in the actuators, or the number of discrete values if
it is a discrete environment.
                
Then, the argument called ``action_space`` defines this action space following the **gymnasium standard**.  
This definition can be discrete or continuous and must be consistent with the previously defined actuators 
(*Sinergym* will show possible inconsistencies).

.. important:: *Sinergym*'s listed environments have a default observation and action variables defined, 
               it is available in `constants.py <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/constants.py>`__.
               However, the users can experiment with this spaces, see :ref:`Changing observation and action spaces`.

*Sinergym* offers the possibility to create **empty action interfaces** too, so that you can take advantage 
of all its benefits instead of using the *EnergyPlus* simulator directly, meanwhile the control is 
managed by **default building model schedulers** as mentioned. For more information, see the example of use 
:ref:`Default building control setting up an empty action interface`.

Normalization flag
===================

The argument called ``flag_normalization`` indicates whether action space specified will be normalized to
``[-1,1]`` or not (only take effect in **continuous** environments). Then, *Sinergym* will use
the real space specified in **action_space** argument or this normalized space depending on
this flag value. This is done in order to make environments more generic in DRL solutions.
*Sinergym* **parse** these values to real action space defined in environment internally before to 
send it to *EnergyPlus* Simulator by the API middleware.

.. important:: The method in charge of parse this values from [-1,1] to real action space if it is required is 
        called ``_action_transform(action)`` in *sinergym/sinergym/envs/eplus_env.py*.
        We always recommend to use the normalization in action space for DRL solutions, since this space is 
        compatible with all algorithms. However, if you are implementing your own rule-based controller 
        and working with real action values, for example, you can deactivate normalization.

.. note:: By default, all *Sinergym*'s environments will have normalization in action space.
        It is possible to specify the **flag_normalization** to false in the constructor argument or
        to change it during the execution using ``env.update_flag_normalization(False)``.

Action mapping
===============

The argument called ``action_mapping`` is only necessary to specify it in **discrete** action spaces. 
It is a dictionary that links an **index** to a specific configuration of values for 
each action variable. The format of this dictionary is:

.. code-block:: python

  action_space = gym.space.Discrete(10)

  action_mapping = {
    0: # <tuple with all action variables values for option 0>
    1: # <tuple with all action variables values for option 1>
    2: # <tuple with all action variables values for option 2>
    # ... 
  }

These tuples must have the same length than the action variables of the environment.

As you can see, some attributes are required depending on the environment is **continuous or discrete**. If
the environment is discrete, ``action_mapping`` is required, if it is specified in a continuous environment will
not take effect. On the other hand, in a continuous environment, it will be created a real and normalized space and 
only one would be the action space depending on ``flag_normalization`` value. If normalization flag is activated
in a discrete environment, will not take effect.

.. image:: /_static/environment_types.png
  :scale: 60 %
  :alt: Attributes depending on environment type.
  :align: center

Environment name
================

The parameter ``env_name`` is used to define the **name of working directory** generation. It is very useful to
difference several experiments in the same environment, for example.

Extra configuration
====================

Some parameters directly associated with the building model and simulator can be set as extra configuration 
as well, such as ``people occupant``, ``timesteps per simulation hour``, ``runperiod``, etc.

Like this **extra configuration context** can grow up in the future, this is specified in ``config_params`` argument.
It is a Python Dictionary where this values are specified. For more information about extra configuration
available for *Sinergym* visit section :ref:`Extra Configuration in Sinergym simulations`.

**************************************
Adding new weathers for environments
**************************************

*Sinergym* includes several weathers covering different types of climate in different areas of the world. 
The aim is to provide the greatest possible diversity for the experiments taking into account certain 
characteristics.

However, you may need or want to include a **new weather** for an experiment. Therefore, this section 
is dedicated to give an explanation of how to do it:

1. Download **EPW** file and **DDY** file in `EnergyPlus page <https://energyplus.net/weather>`__. *DDY* file
   contains information about the location and different design days available for that weather.
2. Both files (*EPW* and *DDY*) must have exactly the same name, being the extension the only difference. 
   They should be placed in the `weathers <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/weather>`__ folder.

That is all! *Sinergym* should be able to adapt ``SizingPeriod:DesignDays`` and ``Site:Location`` fields in building 
model file using *DDY* automatically for that weather.

**************************************
Adding new buildings for environments
**************************************

As we have already mentioned, a user can change the already available environments or even create new environment 
definitions including new climates, action and observation spaces, etc. However, perhaps you want to use a 
**new building model** (*epJSON* file) than the ones we support.

This section is intended to provide information if someone decides to add new buildings for use with *Sinergym*. 
The main steps you have to follow are the next:

1. Add your building file (*epJSON*) to `buildings <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/buildings>`__.
   *EnergyPlus* pretends to work with *JSON* format instead of *IDF* format in their building definitions and simulations. Then,
   *Sinergym* pretends to work with this format from version 2.4.0 or higher directly. You can download a *IDF* file and convert
   to *epJSON* using their **ConvertInputFormat tool** from *EnergyPlus*.
   **Be sure that new epJSON model version is compatible with EnergyPlus version**.

2. Add your own *EPW* file for weather conditions (section :ref:`Adding new weathers for environments`) 
   or use ours in environment constructor. 

3. *Sinergym* will check that observation and action variables specified in environments constructor are 
   available in the simulation before starting. You need to ensure that the variables definition are correct. 

4. Use the environment constructor or register your own environment ID `here <https://github.com/ugr-sail/sinergym/blob/main/sinergym/__init__.py>`__ 
   following the same structure than the demo environment. You will have to specify environment components. 
   We have examples about how to get environment information in :ref:`Getting information about Sinergym environments`.

5. Now, you can use your own environment ID with ``gym.make()`` like our documentation examples.

.. important:: In order to know the available variables, meters, actuators, etc. You can try to do an empty control in the building and look for files
               such as RDD, MDD, MTD or ``data_available.txt`` file generated with *EnergyPlus* API in the output folder by *Sinergym*.

