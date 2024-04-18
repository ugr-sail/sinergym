################
Github Actions
################

This project leverages `Github Actions <https://docs.github.com/en/actions/>`__, 
for continuous integration and deployment, eliminating the need for third-party platforms.

The following procedures are implemented:

*************
Pull Request
*************

- **Python Code format check**: Each pull request is checked for **Pep8** 
  `standard <https://www.python.org/dev/peps/pep-0008/>`__ compliance and 
  `isort <https://github.com/PyCQA/isort>`__ for import sorting. 
  Non-compliance triggers a warning.

- **Code type check**: `pytype <https://github.com/google/pytype>`__ is used in the *Sinergym* module to 
   control function and method I/O types. ``import-error`` types are ignored with ``pytype -d import-error sinergym/``. 
   Type errors halt the workflow until resolved.

- **Documentation checks**: Changes in source documentation trigger a *Sphinx* compilation and 
  spelling check. Errors are raised for compilation issues and unrecognized words. 
  Add unrecognized but correct words to ``docs/source/spelling_wordlist.txt``.

.. Warning:: Sphinx Warning messages are treated as errors.

.. Note:: Sphinx Spelling also checks code docstrings.

- **Testing**: A remote container is built using *Dockerfile* and **Pytest** is executed within it.

- **Repository security**: This workflow identifies differences between source and base in 
  workflows and tests for forked repositories to prevent malicious software. 

.. Note:: Checks can be skipped with ``[ci skip]`` in the commit message. 
          See issue `#161 <https://github.com/ugr-sail/sinergym/issues/161>`__.

************************************
Push main (or merge a pull request)
************************************

The following workflows are executed sequentially:

- **Apply format**: A bot applies necessary format changes (**autopep8** level 2 aggressive and/or **isort** module) 
  in the main branch.

- **Update Documentation build to GitHub pages**: A bot updates the documentation build in the **main** 
  branch when necessary, including spelling checks.

- **Update our Docker Hub repository**: A container with all extra requirements is built and pushed to 
  our `Docker Hub repository <https://hub.docker.com/r/sailugr/sinergym>`__ using the *latest* tag. A 
  minimal container is also included with the *lite* tag.

- **Testing and CodeCov update**: Tests are executed similarly to the pull request event, 
  and the coverage report is uploaded to CodeCov.

********************************
New release created or modified
********************************

- When a **release** is *published* or *edited*, a container is built and updated on Docker Hub 
  with the release tag version.

- Simultaneously, the **PyPi** *Sinergym* repository is updated with the current version tag.

- For **pre-releases**, the PyPi version is updated but not the Docker Hub container.

.. Note:: See `.github/workflows YML files <https://github.com/ugr-sail/sinergym/tree/main/.github/workflows>`__ for the code.

.. Note:: If you forked *Sinergym*, enable Github Action in your project to utilize these functionalities.