**Sinergym**
============

.. include:: pages/introduction.rst

############
Contributing
############

If you are interested in contributing to the development of *Sinergym*, there are several ways you can get involved:

- Reporting bugs or suggesting improvements through `issues <https://github.com/ugr-sail/sinergym/issues>`__.
- Assisting in the development of new features or fixing existing bugs.

Before contributing, please refer to `CONTRIBUTING.md <https://github.com/ugr-sail/sinergym/blob/main/CONTRIBUTING.md>`__ for additional information.

#########################
Projects using *Sinergym*
#########################

The following are some of the projects that utilize *Sinergym*.

- `Demosthen/ActiveRL <https://github.com/Demosthen/ActiveRL>`__
- `VectorInstitute/HV-Ai-C <https://github.com/VectorInstitute/HV-Ai-C>`__
- `rdnfn/beobench <https://github.com/rdnfn/beobench>`__

If you would like to be included in the list, just open a pull request requesting it. Before doing so, please add the following badge to your repository:

.. raw:: html
    :file: ./_templates/sinergym.html

########
Examples
########

If you are new to using *Sinergym*, you will need to perform some initial setup. Please refer to the :ref:`Installation` section for a detailed guide on this process.

Once the setup is complete, we recommend running the examples available in the `examples <https://github.com/ugr-sail/sinergym/tree/main/examples>`__ directory to explore the different features the tool offers.

#################
Citing *Sinergym*
#################

If you use *Sinergym* in your work, please cite our `paper <https://dl.acm.org/doi/abs/10.1145/3486611.3488729>`__::

    @inproceedings{2021sinergym,
      title={*Sinergym*: A Building Simulation and Control Framework for Training Reinforcement Learning Agents}, 
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
   pages/architecture.rst
   pages/environments.rst
   pages/environments_registration.rst
   pages/logging.rst
   pages/output.rst
   pages/rewards.rst
   pages/controllers.rst
   pages/wrappers.rst
   pages/extra-configuration.rst
   pages/deep-reinforcement-learning.rst
   pages/gcloudAPI.rst
   pages/github-actions.rst
   pages/tests.rst


.. toctree::
   :numbered:
   :hidden:
   :caption: Examples

   pages/notebooks/basic_example.nblink
   pages/notebooks/getting_env_information.nblink
   pages/notebooks/sinergym_package.nblink
   pages/notebooks/change_environment.nblink
   pages/notebooks/default_building_control.nblink
   pages/notebooks/wrappers_examples.nblink
   pages/notebooks/personalize_loggerwrapper.nblink
   pages/notebooks/logging_unused_variables.nblink
   pages/notebooks/rule_controller_example.nblink
   pages/notebooks/drl.nblink
   
.. toctree::
   :hidden:
   :caption: API

   pages/API-reference.rst

   