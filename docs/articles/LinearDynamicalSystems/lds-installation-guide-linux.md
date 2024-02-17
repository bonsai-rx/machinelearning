# Installation Guide - Linux

This guide is meant for users to install the package from scratch. *Note* to use this package to run the examples, you must install the necessary Bonsai packages through the Bonsai package manager. Alternatively, you can follow along with the [Examples](../../../examples/README.md) installation guide to learn how to bootstrap the Bonsai and python environments directly from the repo provided.

## Notes on Installing Bonsai on Linux (Ubuntu)

Bonsai on Linux is not as straightforward as installing on Windows and is still being actively tested and optimized. Currently, Bonsai support has only been tested on Ubuntu 22.04. If you wish to try Bonsai on Linux, be sure to read the [Linux Installation Guide](https://github.com/orgs/bonsai-rx/discussions/1101) for important information regarding how to install underlying Bonsai packge dependencies, such as OpenGL, OpenAL, OpenCV, etc., for use in Bonsai workflows.

### Dependencies

To get started, you must install the following tools:

- [Python (v3.10.12)](https://www.python.org/downloads/)
- [dotnet-sdk (v8)](https://dotnet.microsoft.com/en-us/download)
- [Git](https://git-scm.com/downloads)
- [Powershell on Linux](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux?view=powershell-7.4)
- [Bonsai-Rx Linux Environment Template](https://github.com/ncguilbeault/bonsai-linux-environment-template)

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
python3 -m venv .venv
```

3. Create a bonsai environment.

*Note*: this step uses the [bonsai linux environment template tool](https://github.com/ncguilbeault/bonsai-linux-environment-template) for easy creation of linux environments

```cmd
dotnet new bonsaienvl -o .bonsai
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

```python
import lds
```

If no errors appear when `import lds` is called in python, your python environment is ready. Exit python with the following command and move onto the next step:

```python
exit()
```

### Bonsai Environment Setup Guide

1. Activate the bonsai environment and launch bonsai:

```cmd
source .bonsai/activate
bonsai
```

2. Open Bonsai's `Package Manager` from the startup menu and install the `Bonsai.ML.LinearDynamicalSystems` bonsai package from the menu. When installing the package, make sure to enter `Accept` on the prompts when asked to accept *Terms and Conditions* during installation.

You will now have access to `Bonsai.ML.LinearDynamicalSystems` modules from the Bonsai toolbox.

If all of these steps worked for you, head to the section on [Getting Started](lds-getting-started.md).
