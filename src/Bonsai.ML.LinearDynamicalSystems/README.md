# Bonsai.ML - Linear Dynamical Systems

Linear dynamical systems is a library in the Bonsai.ML project. It is designed for making predictions on behavioral data using linear models such as the Kalman Filter. It uses the LDS Python package to run KalmanFilter filtering on real-time movement and trajectory data collected in Bonsai.

## Installation

In order to run this package, users must set up the correct Bonsai and Python environment and install the correct dependencies.

### Dependencies

- [Bonsai-Rx (v2.8.1 or above)](https://bonsai-rx.org/docs/articles/installation.html)
- [Python (v3.10.12 or above)](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

### Bonsai-Rx Package Dependencies:

- Bonsai.ML.LinearDynamicalSystems

### Python Package Dependencies:

- [lds_python](https://github.com/joacorapela/lds_python)

*Note:* this guide assumes you will be using virtual environments.

## Setup Guide

Start by opening up a terminal (powershell on Windows).

### Creating Virtual Environments

1. Create a folder for your project

```cmd
cd ~/Desktop
mkdir LinearDynamicalSystems
cd LinearDynamicalSystems
```

2. Create a python virtual environment.

```cmd
python -m venv .venv # If using windows
```

or

```cmd
python3 -m venv .venv # if using linux
```

3. Create a bonsai environment. When prompted, enter yes to run the powershell setup script.  *Note*: this guide uses the [bonsai linux environment creation tool](https://github.com/ncguilbeault/bonsai-linux-environment-template) for easy creation of linux environments but the Windows method should also work on Linux.

```cmd
dotnet new bonsaienv -o .bonsai # if using Windows
```

or

```cmd
dotnet new bonsaienvl -o .bonsai # if using Linux
```

### Python Environment Setup Guide

1. Activate the python environment

```cmd
.\.venv\Scripts\activate # if using Windows
```

or

```cmd
source .venv/bin/activate # if using Linux
```

2. Install the lds_python package

```cmd
pip install lds_python@git+https://github.com/joacorapela/lds_python@168d4c05bb4b014998c7d3a2a57d143244a44bdd
```

3. Check that the lds_python package has been installed correctly. Launch a python IDE and try importing the package.

```cmd
python # launches python IDE

import lds
```

If no errors appear when `import lds` is called in python, exit python and move onto the next step.

### Bonsai Environment Setup Guide

1. Activate the bonsai environment and launch bonsai

```cmd
.bonsai/Bonsai.exe # if using Windows
```

or

```cmd
# if using Linux environment
source .bonsai/activate
bonsai 
```

2. Install the `Bonsai.ML.LinearDynamicalSystems` bonsai package. Make sure to click Yes on the prompts during installation.

You should now have access to the Bonsai.ML.LinearDynamicalSystems modules inside the Bonsai toolbox.

## General usage

### Activating environments

To use the package, you must activate the python virtual environment in which the lds_python package was installed (see [Python Environment Setup Guide](#python-environment-setup-guide)).

On Linux, it is possible to activate both the python virtual environment and the bonsai environment. The order of activating the environments matters so you must activate the python environment first and then the bonsai environment second.

```cmd
source .venv/bin/activate
source .bonsai/activate
```

### Bonsai.ML.LinearDynamicalSystems Framework

The package works by using the Bonsai.ML.LinearDynamicalSystems package to interact with the python kernel's installation of the lds_python package. The Bonsai package does not perform the computations directly, rather it uses Python.Net and Bonsai.Scripting.Python packages to interop with the underlying python installation. It might be important to know that the computations performed by the Kalman filter for 2D movement tracking of animal behavior are fundamentally happening in Python (using the lds_python package) rather than in Bonsai directly. However, this framework allows us to make use of the powerful tools for online acquisition of data through Bonsai with the powerful Kalman filter tool developed in Python.  

Since this package makes use of both Bonsai (Bonsai.ML.LinearDynamicalSystems) and Python (lds_python), we have provided a suggested framework for the use case of tracking the centroid of an animal in 2D. In the example below, you will see how the Bonsai workflow is built.

[]()
