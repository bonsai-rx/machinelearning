# Introduction

The Bonsai.ML project is a collection of Bonsai packages for implementing machine learning algorithms in Bonsai. The latest LinearDynamicalSystems package is designed for analyzing kinematic data. It uses the LDS Python package to perform inference of kinematics using the Kalman Filter on real-time movement and trajectory data collected in Bonsai.

For installation instructions, head to [Installation Guide](installation-guide.html)

### Overview

The LinearDynamicalSystems package provides an interface to interact with a python installation of the lds_python package. Below is a general guideline for how to use this package to performing online inference of behavioural tracking data, making use of both online behavioural tracking tools (Bonsai) and linear dynamical systems modelling (Python).

### General workflow

The following represents a general workflow for how you would use the LinearDynamicalSystems package. It starts with instantiate the model and subsequently use the model to make predictions about the state of a variable given observations of data.

```mermaid

flowchart LR

    A(["Create Python Runtime"])
    B(["Load LDS Module"])
    C(["Instantiate KFK Model"])
    D(["Create Observation"])
    E(["Perform Inference"])

    A --> B
    B --> C
    C --> D
    D --> E

```

For more information on how to implement these workflows, see [Getting Started](getting-started.html)

