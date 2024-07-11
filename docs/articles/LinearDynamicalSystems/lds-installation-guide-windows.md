# Installation Guide - Windows

This guide is meant for users to install the package from scratch. To run the examples, users must install the additional Bonsai packages required to run each [Example](~/examples/README.md). Some familiarity with the command line or powershell terminal is necessary. This guide has only been tested on Windows 10 and 11, so earlier versions of Windows may or may not work.

### Dependencies

To get started, you must install the following tools:

- [Python (v3.10)](https://www.python.org/downloads/) *Note - in order for Bonsai to pick up the python installation, it must be installed through the standard Windows installer (**not** through the Windows App store) and must be added to the system PATH*
- [dotnet-sdk (v8)](https://dotnet.microsoft.com/en-us/download)
- [Git](https://git-scm.com/downloads)
- [Bonsai-Rx Templates tool](https://www.nuget.org/packages/Bonsai.Templates)

> [!WARNING]
> Be sure to check the specific python version and dotnet-sdk version you have installed, as different version than the ones we recommend may or may not work with this guide.

### Creating Virtual Environments

1. Open up the terminal and create a folder for your project. For example:

```cmd
cd ~\Desktop
mkdir LinearDynamicalSystems
cd .\LinearDynamicalSystems
```

2. Create a python virtual environment inside of your folder.

```cmd
python -m venv .venv
```

> [!TIP]
> If you receive an error that says, `python cannot be found`, check to ensure that python is available on the system path. If you just installed python, it may be necessary to restart the terminal.

3. Create a bonsai environment. When prompted, enter yes to run the powershell setup script.

```cmd
dotnet new bonsaienv
```

> [!TIP]
> If you get an error during this step which says, `Setup.ps1 cannot be loaded because running scripts is disabled`, you need to allow powershell scripts to be executed by users. To do this, you can change the global execution policy by opening a new powershell instance with `Run as Administrator` and use the following command:

```powershell
set-executionpolicy remotesigned
```

Alternatively, you can use the `Setup.cmd` file to setup the bonsai environment without changing the execution policy globally. Run the file by going to the terminal and running:

```cmd
.\.bonsai\Setup.cmd

```

### Python Environment Setup Guide

1. Activate the python environment

```cmd
.\.venv\Scripts\activate
```

2. Install the lds_python package

```cmd
pip install lds_python@git+https://github.com/joacorapela/lds_python@f761c201f3df883503ecb67acef35ba846e3524c
```

If you encounter errors during installation of the lds_python package, you will have to diagnose the issue and install the correct package dependencies manually.

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
.bonsai\Bonsai.exe
```

2. Open Bonsai's `Package Manager` from the startup menu and install the `Bonsai.ML.LinearDynamicalSystems` bonsai package from the menu. When installing the package, make sure to enter `Accept` on the prompts when asked to accept *Terms and Conditions* during installation.

You will now have access to `Bonsai.ML.LinearDynamicalSystems` modules from the Bonsai toolbox.

If all of these steps worked for you, head to the section on [Getting Started](lds-getting-started.md).
