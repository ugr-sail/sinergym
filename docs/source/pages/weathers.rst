############
Weathers
############

Specific information on the climates included in *Sinergym* is provided below, 
independently of the rest of the components that make up the environment to 
be used in each experiment.

.. important:: If you are interested in including new weathers to this framework, 
               please visit section :ref:`Adding new weathers for environments`.

+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
|                        Weather file                         |         Location         |                                       Climate type                                        | M.A.T (ºC) | M.A.H (%) |
+=============================================================+==========================+===========================================================================================+============+===========+
| AUS_NSW.Sydney.947670_IWEC                                  | Sydney, Australia        | Humid subtropical (no dry seasons and hot summers)                                        | 17.9       | 68.83     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| COL_Bogota.802220_IWEC                                      | Bogota, Colombia         | Mediterranean (dry, warm summers and mild winters).                                       | 13.2       | 80.3      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| ESP_Granada.084190_SWEC                                     | Granada, Spain           | Mid-latitude dry semiarid and hot dry periods in summer, but passive cooling is possible. | 14.84      | 59.83     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| FIN_Helsinki.029740_IWEC                                    | Helsinki, Finland        | Moist continental (warm summers, cold winters, no dry seasons).                           | 5.1        | 79.25     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| JPN_Tokyo.Hyakuri.477150_IWEC                               | Tokyo, Japan             | Humid subtropical (mild with no dry season, hot summer).                                  | 8.9        | 78.6      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| MDG_Antananarivo.670830_IWEC                                | Antananarivo, Madagascar | Mediterranean climate (dry warm summer, mild winter).                                     | 18.35      | 75.91     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| PRT_Lisboa.085360_INETI                                     | Lisboa, Portugal         | Dry Summer Subtropical Mediterranean (Warm - Marine).                                     | 16.3       | 74.2      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| USA_AZ_Davis-Monthan.AFB.722745_TMY3                        | Arizona, USA             | Subtropical hot desert (hot and dry).                                                     | 21.7       | 34.9      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| USA_CO_Aurora-Buckley.Field.ANGB.724695_TMY3                | Colorado, USA            | Mid-latitude dry semiarid (cool and dry).                                                 | 9.95       | 55.25     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| USA_IL_Chicago-OHare.Intl.AP.725300_TMY3                    | Illinois, USA            | Humid continental (mixed and humid).                                                      | 9.92       | 70.3      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3             | New York, USA            | Humid continental (mixed and humid).                                                      | 12.6       | 68.5      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3           | Pennsylvania, USA        | Humid continental (cool and humid).                                                       | 10.5       | 66.41     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3 | Washington, USA          | Cool Marine west coastal (warm summer, mild winter, rain all year).                       | 9.3        | 81.1      |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+
| SWE_Stockholm.Arlanda.024600_IWEC                           | Stockholm, Sweden        | Moist continental (warm summer, cold winter, no dry season).                              | 6.43       | 78.42     |
+-------------------------------------------------------------+--------------------------+-------------------------------------------------------------------------------------------+------------+-----------+

*M.A.T: Mean Temperature*,
*M.A.H: Mean Humidity*

.. note:: Weather types according to `DOE's
          classification <https://www.energycodes.gov/development/commercial/prototype_models#TMY3>`__.

.. important:: It can be specified several weathers in the same experiment. *Sinergym* will sample one weather in each episode to use it. For more
               information, visit section :ref:`Weather files`.

