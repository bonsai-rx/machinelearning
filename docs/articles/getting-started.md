# Getting Started

Below is a getting started tutorial for how to use the Bonsai.ML.LinearDynamicalSystems package.

If you followed the installation guide and opted for using virtual environments, follow the [activating environments in Windows](#activating-environments---windows) or [activating environments in Linux](#activating-environments---linux) instructions below. If you opted out of using virtual environments during installation, feel free to skip this section and head directly to the section on [Implementing the KFK Model in Bonsai](#implementing-the-kfk-model-in-bonsai).

### Activating environments - Windows

On windows, all you have to do is activate the python environment, and then launch the Bonsai.exe inside the bonsai environment folder.

```cmd
.venv\Scripts\activate
.bonsai\Bonsai.exe
```

### Activating environments - Linux

If you used the Linux environment creation tool, you can activate you bonsai environment just the same as you would activate your python virtual environment. You can have both the python and the bonsai environments activated at the same time. The order of activating the environments matters so you must activate the python environment first and then the bonsai environment second.

```cmd
# Using the Linux environment creation tool
source .venv/bin/activate
source .bonsai/activate
```

If you did not use the linux environment template but you plan on using linux, you can simply activate the python environment and then run the bonsai application in the folder using mono (see installing Bonsai-Rx on Linux [here](https://github.com/orgs/bonsai-rx/discussions/1101)).

```cmd
source .venv/bin/activate
mono .bonsai/Bonsai.exe
```

### Implementing the KFK Model in Bonsai

Below is a simplified Bonsai workflow that implements the core logic of the `Bonsai.ML.LinearDynamicalSystems` package to run the Kalman Filter Kinematics (KFK) Model for inferring kinematics from observations of behavioral tracking data.

:::workflow
![KFK Model Implementation](~/workflows/KFKModelImplementation.bonsai)
:::

The core logic can be broken down as follows. 

A `CreateRuntime` node is used to initialize a python runtime engine, which gets passed to a `BehaviorSubject` called `PythonEngine`. Bonsai's `CreateRuntime` node should automatically detect the python virtual environment that was used to launch the Bonsai application, otherwise the path to the virtual environment can be specified in the `CreateRuntime` node by setting the `PythonHome` property.

Next, the `PythonEngine` node is passed to a `LoadLDSModule` node which will load the python-bonsai interface to the lds_python package.

Once the LDS module has been initialized, it gets passed to the `CreateKalmanFilterKinematicsModel` node, which instantiates a python instance of the model. Here, you can specify the initialization parameters of the model and provide a `ModelName` parameter that gets used to reference the model in other parts of the Bonsai workflow.

Next, you would take some tracking data (for example, the centroid of an animal or a 2D point), and pass that to a `CreateObservation2D` node which will package the data into a data format that the model can use.

The `Observation` is then passed to a `PerformInference` node, which will use the specified model (given by the ModelName) to infer the state of model and output the inferred behavioural kinematics.

Then, all you have to do is pass your behavior data of interest into the `BehaviorData` subject and the model will start performing inference.

### Further Examples

For examples and demonstrations for how this works, see the [Bonsai - Machine Learning Examples](#) repo.
