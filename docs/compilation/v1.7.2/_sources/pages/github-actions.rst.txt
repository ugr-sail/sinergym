################
Github Actions
################

This project is automatically processed using `Github Action <https://docs.github.com/es/actions/>`__ which allows building continuous integration and continuous deployment pipelines
for testing, releasing and deploying software without the use of third-party websites/platforms.

Currently, we have developed the next procedures for this project:

*************
Pull Request
*************

- **Python Code format check**: Python code format is checked in every pull request following **Pep8** `standard <https://www.python.org/dev/peps/pep-0008/>`__ (Level 2 aggressive). If format is incorrect, a bot will comment in pull request advising that issue and notifying it will be correct merging with main.
- **Documentation Checks**: This action compile documentation *source* in every pull-request, verify if documentation build is successful.

.. warning:: Sphinx Warning messages behave like errors for workflow status.

.. note::

  If you want to ignore *docs/build* files while you are working locally. You can ignore files although files are in repository executing next in local:

    .. code:: sh
        
        $ git ls-files -z docs/build/ | xargs -0 git update-index --assume-unchanged

- **Testing**: There is another action which builds a remote container using *Dockerfile* and executes pytest inner.
- **Repository security**: There is a workflow which compare differences in workflows and tests from source to base. It execute that functionality only in forked repositories in order to prevent malicious software in workflow or ignore tests. Event is *pull_request_target*, this means workflow is checkout from base repository (our main branch) and it cannot be manipulate by third-parties.

************************************
Push main (or merge a pull request)
************************************

- **Apply format autopep8**: A bot generates a commit in main branch applying format changes when it is necessary.
- **Update Documentation build to GitHub pages**: A bot generates a commit in main branch applying new documentation build when it is necessary.
- **Update our Docker Hub repository**: This job builds container with all extra requires and it is pushed to our `Docker Hub repository <https://hub.docker.com/r/alejandrocn7/sinergym>`__ using *latest* tag automatically. It needs format and documentation jobs finish for possible changes.

********************************
New release created or modified
********************************

- When a **release** is *published* or *edited* manually in the repository, there is an action which catches release tag version and uses it to build a container and upload/update on Docker Hub with that tag version.

.. note:: See `.github/workflows YML files <https://github.com/jajimer/sinergym/tree/develop/.github/workflows>`__ to see code used.

.. note:: Wether you have a forked repository from Sinergym, we recommend you to **enable Github Action in your project** in order to take advantage of this functionality in your developments.