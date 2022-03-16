################
Github Actions
################

This project is automatically processed using `Github Action <https://docs.github.com/es/actions/>`__ which allows building continuous integration and continuous deployment pipelines
for testing, releasing and deploying software without the use of third-party websites/platforms.

Currently, we have developed the next procedures for this project:

*************
Pull Request
*************

- **Python Code format check**: Python code format is checked in every pull request following **Pep8** `standard <https://www.python.org/dev/peps/pep-0008/>`__ (Level 2 aggressive) and `isort <https://github.com/PyCQA/isort>`__ to sort imports. 
  If format is incorrect, a bot will comment in pull request advising that issue and notifying it will be correct merging with main.
- **Code type check**: We are using `pytype <https://github.com/google/pytype>`__ in Sinergym module. This let dynamic types in Python like it is usual, but controlling input and output types in functions and methods. This workflow ignore `import-error` type using command `pytype -d import-error sinergym/`.
  For example, **pytype** cannot include google cloud storage module, so this option specification is necessary. If some type error happens, the workflow show error until user fix it.
- **Documentation Checks**: This action checks whether source documentation has been modified in every pull-request. If source documentation has been updated, it will compile documentation with Sphinx and raise errors if exist.
  This workflow checks **vocabulary spelling** too. If you have a mistake and sphinx finds a unknown word, this workflow will return an error. Writing documentation
  about this topic is very possible that you want to use a word that is not in default dictionary. In that case, you have to add that word to `docs/source/spelling_wordlist.txt` (please, respect alphabetical order) and Sphinx-spelling will accept words allocated in the list.

.. warning:: Sphinx Warning messages behave like errors for workflow status.

.. note:: Sphinx Spelling works on code docstring too.

.. note::

  If you want to ignore *docs/build* files while you are working locally. You can ignore files although files are in repository executing next in local:

    .. code:: sh
        
        $ git ls-files -z docs/build/ | xargs -0 git update-index --assume-unchanged

- **Testing**: There is another action which builds a remote container using *Dockerfile* and executes pytest inner.
- **Repository security**: There is a workflow which compare differences in workflows and tests from source to base. It execute that functionality only in forked repositories in order to prevent malicious software in workflow or ignore tests. Event is *pull_request_target*, this means workflow is checkout from base repository (our main branch) and it cannot be manipulate by third-parties.

.. note:: These checks can be skipped in a specific commit writing `[ci skip]` string in commit message. For more information, see issue `#161 <https://github.com/jajimer/sinergym/issues/161>`__.

************************************
Push main (or merge a pull request)
************************************

This workflows will be executed in sequential order:

- **Apply format**: A bot generates a commit in main branch applying format changes when it is necessary (autopep8 2 level aggressive and/or `isort` module).
- **Update Documentation build to GitHub pages**: A bot generates a commit in main branch applying new documentation build when it is necessary (spelling check included here too).
- **Update our Docker Hub repository**: This job builds container with all extra requires and it is pushed to our `Docker Hub repository <https://hub.docker.com/r/alejandrocn7/sinergym>`__ using *latest* tag automatically. It needs format and documentation jobs finish for possible changes.

********************************
New release created or modified
********************************

- When a **release** is *published* or *edited* manually in the repository, there is an action which catches release tag version and uses it to build a container and upload/update on Docker Hub with that tag version.
- At the same time, another job will update the **PyPi** Sinergym repository with its current version tag.

.. note:: See `.github/workflows YML files <https://github.com/jajimer/sinergym/tree/develop/.github/workflows>`__ to see code used.

.. note:: Whether you have a forked repository from Sinergym, we recommend you to **enable Github Action in your project** in order to take advantage of this functionality in your developments.