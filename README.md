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

### Bonsai.ML.Torch
Interfaces with the [TorchSharp](https://github.com/dotnet/TorchSharp) package, a C# wrapper around the torch library. Provides tooling for manipulating tensors, performing linear algebra, training and inference with deep neural networks, and more. 

> [!NOTE]
> Bonsai.ML packages can be installed through Bonsai's integrated package manager and are generally ready for immediate use. However, some packages may require additional installation steps. Refer to the specific package section for detailed installation guides and documentation.

## Development Roadmap
The ultimate goal of the `Bonsai.ML` project is to bring powerful machine learning tools into Bonsai to enable intelligent experimental control. To achieve this, our plan is to incorporate several different packages, models, frameworks, and language integrations. You can follow our progress by going to the [Bonsai ML development roadmap](https://github.com/orgs/bonsai-rx/projects/7).

## Feedback and Contributions
`Bonsai.ML` is an open-source project and we welcome contributions and feedback from the community. If you have any comments or questions, or require assistance, please feel free to open an issue on the [GitHub repo](https://github.com/bonsai-rx/machinelearning). If you would like to contribute to the project, please see our [Contributor Guide](https://bonsai-rx.org/contribute/). By contributing to our project, we also expect you to uphold our community [Code of Conduct](https://bonsai-rx.org/code-of-conduct).

## Acknowledgments

Development of the Bonsai.ML package is supported by the Biotechnology and Biological Sciences Research Council [grant number BB/W019132/1].