# Bonsai - Machine Learning

The **Bonsai.ML** project is a collection of packages designed to integrate machine learning algorithms with Bonsai. This document provides an overview of the available packages and their functionalities.

## Core Packages

- **Bonsai.ML**
    Provides common tools and functionality.

- **Bonsai.ML.Design**
    Provides common tools and functionality for visualizers and editor features.

- **Bonsai.ML.Data**
    Provides common tools and functionality for working with data.

- **Bonsai.ML.Python**
    Provides common tools and functionality for C# packages to interface with Python.

## Available Packages

### Bonsai.ML.LinearDynamicalSystems 
Facilitates inference using linear dynamical systems (LDS). It interfaces with the [lds_python](https://github.com/joacorapela/lds_python) package using the [Bonsai - Python Scripting](https://github.com/bonsai-rx/python-scripting) library.

- **Bonsai.ML.LinearDynamicalSystems.Kinematics**
    Supports the use of the Kalman Filter for inferring kinematic data.

- **Bonsai.ML.LinearDynamicalSystems.LinearRegression** 
    Utilizes the Kalman Filter to perform online Bayesian linear regression.

### Bonsai.ML.LinearDynamicalSystems.Design
Visualizers and editor features for the LinearDynamicalSystems package.

### Bonsai.ML.HiddenMarkovModels
Facilitates inference using Hidden Markov Models (HMMs). It interfaces with the [ssm](https://github.com/lindermanlab/ssm) package using the [Bonsai - Python Scripting](https://github.com/bonsai-rx/python-scripting) library.

- **Bonsai.ML.HiddenMarkovModels.Observations**
    Provides functionality for specifying different types of observations.

- **Bonsai.ML.HiddenMarkovModels.Transitions**
    Provides functionality for specifying different types of transition models.

### Bonsai.ML.HiddenMarkovModels.Design
Visualizers and editor features for the HiddenMarkovModels package.

### Bonsai.ML.NeuralDecoder
Enables online neural decoding of spike sorted or clusterless neural activity. It interfaces with the [bayesian-neural-decoder](https://github.com/ncguilbeault/bayesian-neural-decoder) package using the [Bonsai - Python Scripting](https://github.com/bonsai-rx/python-scripting) library. The neural decoder consists of a bayesian state-space point process model to decode sorted spikes or clusterless neural activity. The technical details describing the models implementation are described in: Denovellis, E.L., Gillespie, A.K., Coulter, M.E., et al. Hippocampal replay of experience at real-world speeds. eLife 10, e64505 (2021). https://doi.org/10.7554/eLife.64505.

### Bonsai.ML.NeuralDecoder.Design
Visualizers for the Neural Decoder package.

> [!NOTE]
> Bonsai.ML packages can be installed through Bonsai's integrated package manager and are generally ready for immediate use. However, some packages may require additional installation steps. Refer to the specific package section for detailed installation guides and documentation.

## Acknowledgments

Development of the Bonsai.ML package is supported by the Biotechnology and Biological Sciences Research Council [grant number BB/W019132/1].