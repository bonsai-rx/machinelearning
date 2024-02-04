# Overview

The LinearDynamicalSystems package provides a Bonsai interface to interact with a python installation of the [lds_python](https://github.com/joacorapela/lds_python) package. Below is a general workflow for implementing the core LinearDynamicalSystems package.

### General workflow

The general workflow for using the LinearDynamicalSystems package starts with creating the python runtime followed by loading the [lds_python](https://github.com/joacorapela/lds_python) module. After this, you create the model and perform inference given new observations of data.

```mermaid

flowchart LR

    A(["Create Python Runtime"])
    B(["Load LDS Module"])
    C(["Instantiate Model"])
    D(["Create Observation"])
    E(["Perform Inference"])

    A --> B
    B --> C
    C --> D
    D --> E

```

The LinearDynamicalSystems package requires additional installation steps before using. See the [Installation Guide - Windows](lds-installation-guide-windows.md) or the [Installation Guide - Linux](lds-installation-guide-linux.md) sections for more information.