# Contributing to Sinergym

Thank you for considering contributing to Sinergym! 🎉  
Your contributions can fall into two main categories:

## 🚀 Proposing a New Feature
1. **Create an issue** using our issue template to propose your feature.
2. **Discuss the design and implementation** with the maintainers.
3. Once approved, **implement the feature** and submit a Pull Request.

## 🛠️ Fixing a Bug or Implementing an Existing Feature
1. Browse open issues: [Sinergym Issues](https://github.com/ugr-sail/sinergym/issues).
2. **Claim an issue** by commenting on it.
3. If you need more context, feel free to ask—we're happy to help!

## 🔄 Submitting a Pull Request (PR)
- **Target the `main` branch** of [Sinergym](https://github.com/ugr-sail/sinergym).
- **Follow the Pull Request template** (it will appear in the PR textbox).
- **Reference the issue number** in your PR description using `Fixes #123` to auto-close it when merged.
- If your work is still in progress, use a **Draft PR** ([learn more](https://github.blog/2019-02-14-introducing-draft-pull-requests/)).

**Need help with PRs?** Check out:
- [How to create a GitHub Pull Request (Stack Overflow)](http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request)
- [GitHub Help: Creating a PR](https://help.github.com/articles/creating-a-pull-request/)

---

# 📌 Creating Issues
- Use our **GitHub issue templates** when possible.
- If your issue doesn't fit any template, submit a **blank issue** with a clear and concise description.

---

# 🖥️ Developing Sinergym

## 💻 **Option 1: Local Installation**
1. **Clone the repository:**
    ```bash
    git clone https://github.com/ugr-sail/sinergym.git
    cd sinergym/
    ```
2. **Install in "extra" mode** (for docs, tests, DRL algorithms, etc.):
    ```bash
    pip install -e .[extra]
    ```
    or using poetry (install all developer dependencies directly):
    ```bash
    poetry install --no-interaction 
    ```

3. **Install EnergyPlus and include EnergyPlus Python API in Python path** (see [INSTALL.md](https://github.com/ugr-sail/sinergym/blob/main/INSTALL.md)).

## 🐳 **Option 2: Using a Docker Container (Recommended)**
😎 **Why?** No need to manage Python versions, dependencies, or EnergyPlus installation.

1. If using **Visual Studio Code**, use the [Remote Containers extension](https://code.visualstudio.com/docs/remote/containers) for an easy setup.
2. Alternatively, build the container manually using the provided `Dockerfile`.

---

# 📖 Documentation Contributions
- Ensure the documentation **builds successfully** with Sphinx:
    ```bash
    cd docs && make spelling && make html
    ```
- If a word isn't recognized, add it to `docs/sources/spelling_wordlist.txt` (**in alphabetical order**).

## 📜 **Docstrings in code**
Follow the **Google-style docstrings** ([Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)):

```python
def my_function(arg1: type1, arg2: type2) -> return_type:
    """Brief summary of the function.

    Args:
        arg1 (type1): Description of arg1.
        arg2 (type2): Description of arg2.

    Returns:
        return_type: Description of return value.
    """
    ...
    return my_variable
```

💡 **Tip**: Use the [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) extension in **VS Code** to automatically generate docstrings in the correct format.

---

# 🎨 Code Style & Formatting

## 📏 **Coding Standards**
> **Note**: Configuration for these tools is automatically applied and included in the pyproject.toml definition.
- **Code formatting**: Use [black](https://black.readthedocs.io/en/stable/) (max **88** characters per line).
- **Import sorting**: Use [ruff](https://docs.astral.sh/ruff/).
- **Type checking**: Run [pyright](https://github.com/microsoft/pyright).

📝 **Before submitting a PR, ensure your code passes:** 
- `black --check .`
- `ruff check .`
- `pyright sinergym/ `
- `pytest tests/ -vvv`

📎 **Git Hooks Setup**: To ensure code quality standards are met before committing, we recommend setting up the provided git hooks:
```bash
chmod +x hooks/*
git config core.hooksPath hooks
```
This will automatically run `black`, `ruff`, and `pyright` (with project's specific settings) checks before each commit.

📋 Workflows in the pull request will check it, in any case.

---

# ❗ Final Checklist Before Submitting a PR  
📝 **Before pushing your code, make sure it meets the following criteria:**  

## 📌 **Types of Changes**  
Indicate the type of change your PR introduces by marking an `x` in the relevant box:

- 🐛 **Bug fix** (non-breaking change that fixes an issue)  
- ✨ **New feature** (non-breaking change that adds functionality)  
- 💥 **Breaking change** (change that affects existing functionality)  
- 📖 **Documentation** (updates to documentation)  
- 🚀 **Improvement** (enhancing an existing feature)  
- 🔄 **Other** (please specify)  

## ✔️ **General PR Checklist**  
<!--- Check all the following points before submitting your PR. If you're unsure, feel free to ask! -->
- ✅ I have read the [CONTRIBUTING.md](https://github.com/ugr-sail/sinergym/blob/main/CONTRIBUTING.md) guide (**required**).  
- ✅ My changes require updates to the documentation.  
- ✅ I have updated the necessary documentation accordingly.  
- ✅ I have added or updated the necessary tests.  
- ✅ I have reformatted the code using **`black`**.  
- ✅ I have sorted imports using **`ruff`**.  
- ✅ If I modified documentation, I verified that **`cd docs && make spelling && make html`** passes.  
- ✅ I ensured that **`pytest tests/ -vv`** runs successfully (**required**).  
- ✅ I checked that **`pyright sinergym/`** runs successfully (**required**).  

---

💡 **Need help?** If you're unsure about anything, feel free to ask in your PR or open an issue.  
Happy coding! 🚀
