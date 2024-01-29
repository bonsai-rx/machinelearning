# Installation

To install, check out the setup guide that is specific to the OS you are using.

## Setup Guide - Windows

Setting up the package requires the following dependencies on Windows.

### Dependencies

To get started, you must install the following tools. See the installation instructions for each tool to get started with each.

- [Python (v3.10.12)](https://www.python.org/downloads/)
- [dotnet-sdk (v8)](https://dotnet.microsoft.com/en-us/download)
- [Git](https://git-scm.com/downloads)
- [Bonsai-Rx Templates tool](https://www.nuget.org/packages/Bonsai.Templates)

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

If no errors appear when `import lds` is called in python, your python environment is ready. Exit python with the following command and move onto the next step:

```python
exit() # exit the python IDE
```

### Bonsai Environment Setup Guide

1. Launch bonsai:

```cmd
.bonsai/Bonsai.exe
```

2. Open Bonsai's `Package Manager` from the startup menu and install the `Bonsai.ML.LinearDynamicalSystems` bonsai package from the menu. When installing the package, make sure to enter `Yes` on the prompts when asked to accept *Terms and Conditions* during installation.

You will now have access to `Bonsai.ML.LinearDynamicalSystems` modules from the Bonsai toolbox.

If all of these steps worked for you, head to the [Getting Started](getting-started.html) section to get started using the package in Bonsai.

## Setup Guide - Linux

Installing the packge on Linux requires a few additional steps compred to Windows (see guide to installing Bonsai-Rx on Linux [here](https://github.com/orgs/bonsai-rx/discussions/1101)).

### Dependencies

To get started, you must install the following tools. See the installation instructions for each tool to get started with each.

- [Python (v3.10.12)](https://www.python.org/downloads/)
- [dotnet-sdk (v8)](https://dotnet.microsoft.com/en-us/download)
- [Git](https://git-scm.com/downloads)
- [Powershell on Linux](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux?view=powershell-7.4)
- [Bonsai-Rx Linux Environment Template](https://github.com/ncguilbeault/bonsai-linux-environment-template)
- [Mono](https://www.mono-project.com/docs/getting-started/install/linux/)

### Creating Virtual Environments

1. Create a folder for your project

```cmd
cd ~/Desktop
mkdir LinearDynamicalSystems
cd LinearDynamicalSystems
```

2. Create a python virtual environment.

```cmd
python3 -m venv .venv
```

3. Create a bonsai environment. 

*Note*: this step uses the [bonsai linux environment template tool](https://github.com/ncguilbeault/bonsai-linux-environment-template) for easy creation of linux environments

```cmd
dotnet new bonsaienvl -o .bonsai # if using Linux environment creation tool
```

When prompted, enter yes to run the powershell setup script.

### Python Environment Setup Guide

1. Activate the python environment

```cmd
source .venv/bin/activate
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

```cmd
import lds
```

If no errors appear when `import lds` is called in python, your python environment is ready. Exit python with the following command and move onto the next step:

```cmd
exit()
```

### Bonsai Environment Setup Guide

1. Activate the bonsai environment and launch bonsai:

```cmd
# if using Linux environment creation tool
source .bonsai/activate
bonsai
```

2. Open Bonsai's `Package Manager` from the startup menu and install the `Bonsai.ML.LinearDynamicalSystems` bonsai package from the menu. When installing the package, make sure to enter `Yes` on the prompts when asked to accept *Terms and Conditions* during installation.

You will now have access to `Bonsai.ML.LinearDynamicalSystems` modules from the Bonsai toolbox.

Head to the [Getting Started](getting-started.html) section to get started using the package in Bonsai.
