Welcome to **sinergym**!
========================

.. include:: pages/introduction.rst

############
Contributing
############

For all those interested in improving Sinergym, there are always improvements to be made. 
You can check `issues <https://github.com/ugr-sail/sinergym/issues>`__ in the repo.

If you want to contribute, please read `CONTRIBUTING.md <https://github.com/ugr-sail/sinergym/blob/main/CONTRIBUTING.md>`__ first.

############
Examples
############

The examples can be run if you have your computer or container properly configured (see :ref:`Installation` section) from our notebooks hosted in the `examples <https://github.com/ugr-sail/sinergym/tree/main/examples>`__ folder of the official Sinergym repository.

################
Citing Sinergym
################

If you use Sinergym in your work, please cite our `paper <https://dl.acm.org/doi/abs/10.1145/3486611.3488729>`__::

    @inproceedings{2021sinergym,
      title={Sinergym: A Building Simulation and Control Framework for Training Reinforcement Learning Agents}, 
      author={Jiménez-Raboso, Javier and Campoy-Nieves, Alejandro and Manjavacas-Lucas, Antonio and Gómez-Romero, Juan and Molina-Solana, Miguel},
      year={2021},
      isbn = {9781450391146},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3486611.3488729},
      doi = {10.1145/3486611.3488729},
      booktitle = {Proceedings of the 8th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
      pages = {319–323},
      numpages = {5},
    }

.. toctree::
   :numbered: 
   :hidden:
   :caption: Start Here

   pages/installation.rst
   pages/usage-example.rst


.. toctree::
   :numbered:
   :hidden:
   :caption: sinergym

   pages/buildings.rst
   pages/weathers.rst
   pages/environments.rst
   pages/rewards.rst
   pages/controllers.rst
   pages/wrappers.rst
   pages/extra-configuration.rst
   pages/output.rst
   pages/deep-reinforcement-learning.rst
   pages/gcloudAPI.rst
   pages/github-actions.rst
   pages/tests.rst


.. toctree::
   :numbered:
   :hidden:
   :caption: Examples

   pages/notebooks/basic_example.nblink
   pages/notebooks/change_environment.nblink
   pages/notebooks/default_building_control.nblink
   pages/notebooks/wrappers_examples.nblink
   pages/notebooks/personalize_loggerwrapper.nblink
   pages/notebooks/rule_controller_example.nblink
   pages/notebooks/drl.nblink
   pages/notebooks/MLflow_example.nblink
   pages/notebooks/TensorBoard_example.nblink

.. toctree::
   :hidden:
   :caption: API

   pages/API-reference.rst

   