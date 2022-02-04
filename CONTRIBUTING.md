## Contributing to Sinergym

If you are interested in contributing to Sinergym, your contributions will fall
into two categories:
1. You want to propose a new Feature and implement it
    - Create an issue about your intended feature (use our issue template), and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/jajimer/sinergym/issues
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/jajimer/sinergym main branch. Please, follow our pull request template for this purpose
(it will appear in text box so you only have to complete it).

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## Create Issues

Please, follow the issue templates whenever possible (GitHub interface will offer you those templates). If your issue does not fit any of the templates, then you can generate a blank issue and provide the information as clearly and concisely as possible.

## Developing Sinergym

To develop Sinergym on your machine, here are some alternatives:

- Local computer

    1. Clone a copy of Sinergym from source (your forked repository):

    ```bash
    git clone https://github.com/jajimer/sinergym.git
    cd sinergym/
    ```

    2. Install Sinergym in extra mode, with support for building the docs, running tests, execute DRL algorithms, etc:

    ```bash
    pip install -e .[extra]
    ```

    3. Install Energyplus and BCVTB into your computer (see [README.md](https://github.com/jajimer/sinergym/blob/main/README.md) for more information about this).

- Local docker container (**recommended**)

    1. This is our recommendation, since you have not to care of Python version, dependencies, Energyplus engine, etc.
    2. If you are using [Visual Studio Code](https://code.visualstudio.com/) you can build a container and develop with Github from there using [remote container extension](https://code.visualstudio.com/docs/remote/containers) (very comfortable).
    3. If you don't want to use that editor extension, you can build a container using Dockerfile traditionally and init github credentials manually into container.

## Codestyle

- We are using [autopep8 codestyle](https://github.com/hhatto/autopep8) (max line length of 79 characters) 2 level aggressive.
- We are using [isort](https://github.com/PyCQA/isort) to sort imports in our code.

**Please run `autopep8`** to reformat your code, if you are using [Visual Studio Code](https://code.visualstudio.com/) editor you can auto-format code on save file automatically: https://code.visualstudio.com/docs/python/editing#_formatting.

Please document each function/method using docstring format:

```python

def my_function(arg1: type1, arg2: type2) -> returntype:
    """[summary]

    Args:
        arg1 (type1): [description]
        arg2 (type2): [description]

    Returns:
        returntype: [description]
    """
    ...
    return my_variable
```

Note: [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) extension could be very interesting for you in order to follow this standard.
