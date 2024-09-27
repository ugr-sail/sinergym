################
Github Actions
################

This project is automatically processed using `Github Action <https://docs.github.com/es/actions/>`__ which allows building continuous integration and continuous deployment pipelines
for testing, releasing and deploying software without the use of third-party websites/platforms.

Currently, we have developed the next procedures for this project:

- **Python Code format**: Python code format is checked in every pull request following **Pep8** `standard <https://www.python.org/dev/peps/pep-0008/>`__ (Level 2 aggressive).
- **Testing**: There is another action which builds a remote container using *Dockerfile* and executes pytest inner.
- When a **release** is *published* or *edited* manually in the repository, there is an action which catches release tag version and uses it to build a container and upload/update on Docker Hub with that tag version.
- **Docs Checks and Update Github-Pages**: This action compile documentation *source* in every pull-request, verify if documentation built is updated and commit and push automatically from a bot account.
  If you want to ignore *docs/build* files while you are working. You can ignore files although files are in repository executing next in local:

    .. code:: sh
        
        $ git ls-files -z docs/build/ | xargs -0 git update-index --assume-unchanged

- **Update our Docker Hub repository**: When main branch is **pushed** (it is always from a pull request merge), there is a workflow which build container with all extra requires and it is pushed to our `Docker Hub repository <https://hub.docker.com/r/alejandrocn7/sinergym>`__ using *latest* tag automatically.

.. note:: See `.github/workflows YML files <https://github.com/jajimer/sinergym/tree/develop/.github/workflows>`__ to see code used.