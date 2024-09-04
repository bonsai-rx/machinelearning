# Bonsai - Machine Learning

The **Bonsai.ML** project is a collection of packages designed to integrate machine learning algorithms with Bonsai. This document provides an overview of the available packages and their functionalities.

## Core Packages

- **Bonsai.ML**
    Provides common tools and functionality.

- **Bonsai.ML.Python**
    Provides common tools and functionality for C# packages to interface with Python.

## Available Packages

### Bonsai.ML.LinearDynamicalSystems 
Facilitates inference using linear dynamical systems (LDS). It interfaces with the [lds_python](https://github.com/joacorapela/lds_python) package using the [Bonsai - Python Scripting](https://github.com/bonsai-rx/python-scripting) library.

- **Bonsai.ML.LinearDynamicalSystems.Kinematics**
    Supports the use of the Kalman Filter for inferring kinematic data.

- **Bonsai.ML.LinearDynamicalSystems.LinearRegression** 
    Utilizes the Kalman Filter to perform online Bayesian linear regression.

### Bonsai.ML.HiddenMarkovModels
Facilitates inference using Hidden Markov Models (HMMs). It interfaces with the [ssm](https://github.com/lindermanlab/ssm) package using the [Bonsai - Python Scripting](https://github.com/bonsai-rx/python-scripting) library.

- **Bonsai.ML.HiddenMarkovModels.Observations**
    Provides functionality for specifying different types of observations.

- **Bonsai.ML.HiddenMarkovModels.Transitions**
    Provides functionality for specifying different types of transition models.
  
### Bonsai.ML.Visualizers
Graphing and plotting library for visualizing data.

> [!NOTE]
> Bonsai.ML packages can be installed through Bonsai's integrated package manager and are generally ready for immediate use. However, some packages may require additional installation steps. Refer to the specific package section for detailed installation guides and documentation.

## Acknowledgments

Development of the Bonsai.ML package is supported by the Biotechnology and Biological Sciences Research Council [grant number BB/W019132/1].