# Getting Started

The `Bonsai.ML.PointProcessDecoder` package provides a Bonsai interface to the [PointProcessDecoder](https://github.com/ncguilbeault/PointProcessDecoder) package used for decoding neural activity (point processes), and relies on the `Bonsai.ML.Torch` package for tensor operations.

## Installation

The package can be installed by going to the bonsai package manager and installing the `Bonsai.ML.PointProcessDecoder`. Additional installation steps are required for installing the CPU or GPU version of the `Bonsai.ML.Torch` package. See the [Torch installation guide](../Torch/torch-overview.md) for more information.

## Package Overview

The `PointProcessDecoder` package is a C# implementation of a Bayesian state space point process decoder inspired by the [replay_trajectory_classification repository](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) from the Eden-Kramer Lab. It can decode latent state observations from spike-train data or clusterless mark data based on point processes using Bayesian state space models.

For more detailed information and documentation about the model, please see the [PointProcessDecoder repo](https://github.com/ncguilbeault/PointProcessDecoder).

## Bonsai Implementation

The following workflow showcases the core functionality of the `Bonsai.ML.PointProcessDecoder` package.

:::workflow
![Point Process Decoder Implementation](~/workflows/PointProcessDecoder.bonsai)
:::

The `CreatePointProcessModel` node is used to define a model and configure it's parameters. For details on model configuration, please see the [PointProcessDecoder documentation](https://github.com/ncguilbeault/PointProcessDecoder). Crucially, the user must specify the `Name` property in the `Model Parameters` section, as this is what allows you to reference the specific model in the `Encode` and `Decode` nodes, the two main methods that the model will use.

During encoding, the user passes in a tuple of `Observation`s and `SpikeCounts`. Observations are variables that the user measures (for example, the animal's position), represented as a (M, N) tensor, where M is the number of samples in a batch and N is the dimensionality of the observations. Spike counts are the data you will use for decoding. For instance, spike counts might be sorted spike data or clusterless marks. If the data are sorted spiking units, then the `SpikeCounts` tensor will be an (M, U) tensor, where U is the number of sorted units. If the data are clusterles marks, then the `SpikeCounts` tensor will be an (M, F, C) tensor, where F is the number of features computed for each mark (for instance, the maximum spike amplitude across electrodes), and C is the number of independant recording channels (for example, individual tetrodes). Internally, the model fits the data as a point process, which will be used during decoding.

Decoding is the process of taking just the `SpikeCounts` and inferring what is the latent `Observation`. To do this, the model uses a bayesian state space model to predict a posterior distribution over the latent observation space using the information contained in the spike counts data. The output of the `Decode` node will provide you with an (M x D*) tensor, with D* representing a discrete latent state space determined by the parameters defined in the model.