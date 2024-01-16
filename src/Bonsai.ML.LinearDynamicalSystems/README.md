# Bonsai.ML - Linear Dynamical Systems

Linear dynamical systems is a library in the Bonsai.ML project. It is designed for making predictions on behavioral data using linear models such as the Kalman Filter. It uses the LDS Python package to run KalmanFilter filtering on real-time movement and trajectory data collected in Bonsai.

## How it works

The package works by using C# code to interact with the LDS Python package for performing 2D movement tracking of animal behavior. In order to run this package, users must set up the correct Bonsai and Python environment and install the correct dependencies.

## Dependencies

- Bonsai-Rx (v2.8.1 or greater)
- Python 3.10

#### Bonsai-Rx Package Dependencies:

- Bonsai.Scripting.Python
- Bonsai.Visualizers

#### Python Package Dependencies:

- Numpy
- Tensorflow

## Getting Started

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

3. Create a bonsai environment. When prompted, enter yes to run the powershell setup script

```cmd
dotnet new bonsaienvl -o .bonsai
```

## Python Setup Guide

1. Activate the python environment

```cmd
source .venv/bin/activate
```

2. Install the lds_python package

```cmd
pip install lds_python@git+https://github.com/joacorapela/lds_python@168d4c05bb4b014998c7d3a2a57d143244a44bdd
```

3. Check that the lds_python package has been installed correctly. Launch a python IDE and try importing the package.

```cmd
python
import lds
```

If no errors appear when `import lds` is called in python, move to the next step.

### Bonsai Environment Setup Guide

1. Activate the bonsai environment and launch bonsai

```cmd
source .bonsai/activate
bonsai
```

2. Select the Bonsai package manager and install the following  packages:

   * `Bonsai - Python Scripting Library`
   * `Bonsai - System Library`
   * `Bonsai - Vision Library`
   * `Bonsai - Vision Design Library`
   * `Bonsai - Scripting Library`
3. Install the `Bonsai.ML.LinearDynamicalSystems` bonsai package.

### Usage

1. Activate the python environment and then the bonsai environment. The order of activating the environments matters so you must activate first the python environment and then the bonsai environment.

```cmd
source .venv/bin/activate
source .bonsai/activate
```

2. Create a folder to store workflows.

```cmd
mkdir workflows
```

3. Create a new workflow in Bonsai. 

You should now have access to the Bonsai.ML.LinearDynamicalSystems modules inside your workflow.
