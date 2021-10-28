################
Github Actions
################

This project is automatically processed using `Github Action <https://docs.github.com/es/actions/>`__ which allows building continuous integration and continuous deployment pipelines
for testing, releasing and deploying software without the use of third-party websites/platforms.

Currently, we have developed the next procedures for this project:

- **Python Code format**: Python code format is checked in every pull request following **Pep8** `standard <https://www.python.org/dev/peps/pep-0008/>`__ (Level 2 aggressive).
- **Installation, Testing and Updating**: There is another action which builds a remote container using *Dockerfile*, executes pytest inner and, if tests have been passed, uploads our **Docker Hub** repository using *latest* tag.
- When a **release** is *published* or *edited* manually in the repository, there is an action which catches release tag version and uses it to build a container and upload/update on Docker Hub with that tag version.

.. note:: See `.github/workflows YML files <https://github.com/jajimer/sinergym/tree/develop/.github/workflows>`__ to see code used.