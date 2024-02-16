# Installation Guide - Windows

This guide is meant for users to install the package from scratch. *Note* to use this package to run the examples, you must install the additional Bonsai packages required or follow the [Examples](../../../examples/README.md) guide to bootstrap the specific bonsai and python environments.

### Dependencies

To get started, you must install the following tools:

- [Python (v3.10.12)](https://www.python.org/downloads/)
- [dotnet-sdk (v8)](https://dotnet.microsoft.com/en-us/download)
- [Git](https://git-scm.com/downloads)
- [Bonsai-Rx Templates tool](https://www.nuget.org/packages/Bonsai.Templates)
- [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)

*Note* be sure to check the python version and dotnet-sdk version to make sure you have the correct version of the tool installed, otherwise the steps in this guide may not work.

### Creating Virtual Environments

1. Create a folder for your project

```cmd
cd ~/Desktop
mkdir LinearDynamicalSystems
cd LinearDynamicalSystems
```

2. Create a python virtual environment.

```cmd
python -m venv .venv
```

3. Create a bonsai environment. When prompted, enter yes to run the powershell setup script.

```cmd
dotnet new bonsaienv -o .bonsai
```

### Python Environment Setup Guide

1. Activate the python environment

```cmd
.\.venv\Scripts\activate
```

2. Install the lds_python package

```cmd
pip install lds_python@git+https://github.com/joacorapela/lds_python@168d4c05bb4b014998c7d3a2a57d143244a44bdd
```

If you encounter errors during installation of the lds_python package, you will have to diagnose the issue and install the correct packge dependencies manually.

3. Check that the lds_python package has been installed correctly. Launch a python IDE:

```cmd
python
```

4. Check that you can correctly import the package:

```python
import lds
```

If no errors appear when `import lds` is called in python, your python environment is ready. Exit python with the following command:

```python
exit()
```

Move onto the next step.

### Bonsai Environment Setup Guide

1. Launch bonsai:

```cmd
.bonsai/Bonsai.exe
```

2. Open Bonsai's `Package Manager` from the startup menu and install the `Bonsai.ML.LinearDynamicalSystems` bonsai package from the menu. When installing the package, make sure to enter `Accept` on the prompts when asked to accept *Terms and Conditions* during installation.

You will now have access to `Bonsai.ML.LinearDynamicalSystems` modules from the Bonsai toolbox.

If all of these steps worked for you, head to the section on [Getting Started](lds-getting-started.md).
