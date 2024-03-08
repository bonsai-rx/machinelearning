# Introduction

The Bonsai.ML project is a collection of Bonsai packages for adding machine learning algorithms in Bonsai. Below you will find the list of packages (and the included subpackages) currently available within the Bonsai.ML collection.

* Bonsai.ML - provides core functionality across all Bonsai.ML packages.
* [Bonsai.ML.LinearDynamicalSystems](./articles/LinearDynamicalSystems/lds-overview.md) - package for performing inference of linear dynamical systems. Interfaces with the [lds_python](https://github.com/joacorapela/lds_python) package. Requires additional installation steps (see [here](./articles/LinearDynamicalSystems/lds-overview.md)).
  - *Bonsai.ML.LinearDynamicalSystems.Kinematics* - subpackage included in the LinearDynamicalSystems package which supports using the Kalman Filter to infer kinematic data.
* Bonsai.ML.Visualizers - provides a set of visualizers for dynamic graphing/plotting.

> [!NOTE]
> Bonsai.ML packages are installed through Bonsai's integrated package manager and are typically available for use immediately. However, certain packages may require additional steps for installation. See the dedicated package section for specific guides and documentation.
