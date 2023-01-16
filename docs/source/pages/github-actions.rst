################
Github Actions
################

This project is automatically processed using `Github Actions <https://docs.github.com/es/actions/>`__, 
a tool to build continuous integration and continuous deployment pipelines
for testing, releasing and deploying software without the use of third-party 
websites/platforms.

Currently, we have developed the next procedures for this project:

*************
Pull Request
*************

- **Python Code format check**: Python code format is checked in every pull 
  request following **Pep8** `standard <https://www.python.org/dev/peps/pep-0008/>`__ 
  (Level 2 aggressive) and `isort <https://github.com/PyCQA/isort>`__ to sort imports. 
  In case the code does not follow the standar, a warning will rise during the workflow execution.

- **Code type check**: We are using `pytype <https://github.com/google/pytype>`__ in 
  *Sinergym* module. This check controls input and output types in functions and methods. This workflow ignores `import-error` 
  type using command `pytype -d import-error sinergym/`.
  For example, **pytype** cannot include google cloud storage module, so this option 
  specification is necessary. If some type error happens, the workflow shows an error until the
  user fixes it.

- **Documentation checks**: This action checks whether source documentation has been 
  modified in every pull-request. If source documentation has been updated, it will 
  **compile** documentation with *Sphinx* and raise errors if they exist.
  This workflow checks **vocabulary spelling** too. If you have a mistake and *Sphinx* 
  finds an unknown word, this workflow will return an error. In case you want to use a word that is not in the default 
  dictionary, please add that word to `docs/source/spelling_wordlist.txt` 
  (please, respect alphabetical order) because Sphinx-spelling accepts words allocated 
  in that list.

.. Warning:: Sphinx Warning messages behave like errors for workflow status.

.. Note:: Sphinx Spelling works on code docstring too.

- **Testing**: It is an action that builds a remote container using *Dockerfile* and executes **Pytest** inside that container. It is a remote container because it is built in Github, just for testing purposes.

- **Repository security**: This workflow identifies differences between source and base in workflows and tests. It executes that functionality only in forked repositories in order to **prevent malicious software** in the workflow, for instances, attemps to ignore tests. 
  The event is *pull_request_target*, this means the workflow is checkout from base repository 
  (our main branch) and it cannot be manipulated by third-parties.

.. Note:: These checks can be skipped in a specific commit writing `[ci skip]` string 
          in commit message. For more information, see issue 
          `#161 <https://github.com/ugr-sail/sinergym/issues/161>`__.

************************************
Push main (or merge a pull request)
************************************

These workflows will be executed in sequential order:

- **Apply format**: A bot generates a commit in the main branch applying 
  format changes when it is necessary (**autopep8** 2 level aggressive 
  and/or **isort** module).

- **Update Documentation build to GitHub pages**: A bot generates a commit 
  in **main** branch applying new documentation build when it is necessary 
  (spelling check included here too) in a folder called **docs/compilation**. 
  The version control ignores the default folder name *build*.

- **Update our Docker Hub repository**: This job builds a container with all extra 
  requirements and it is pushed to our 
  `Docker Hub repository <https://hub.docker.com/r/sailugr/sinergym>`__ 
  using *latest* tag automatically. This update is execured only when the preivous format and documentation workflows have succesfully finished.

********************************
New release created or modified
********************************

- When a **release** is *published* or *edited* manually in the repository, 
  an action catches the release tag version and it uses it to build 
  a container and upload/update on Docker Hub with that tag version.

- At the same time, another job will update the **PyPi** *Sinergym* repository 
  with its current version tag.

.. Note:: See `.github/workflows YML files <https://github.com/ugr-sail/sinergym/tree/main/.github/workflows>`__ 
          to see the code we use.

.. Note:: If you forked the repository from *Sinergym*, we recommend you to
          **enable Github Action in your project** in order to take advantage of 
          this functionality in your developments.

.. Note:: Currently, the workflows explained above upload two containers. A 
          container with **all extra packages** and a container with **minimal**
          installation.
