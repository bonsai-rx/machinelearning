# Overview

The `Bonsai.ML.PointProcessDecoder` package provides a Bonsai interface to the [PointProcessDecoder](https://github.com/ncguilbeault/PointProcessDecoder) package, which is a C# implementation of a Bayesian state-space point process decoder inspired by the [replay_trajectory_classification](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) algorithm. It is designed to decode behaviour or observations from spike-train data or clusterless data using point processes and Bayesian state-space models. 

For more information and documentation about the model and how it works, please see the [PointProcessDecoder repo](https://github.com/ncguilbeault/PointProcessDecoder).

## Installation

The package can be installed by going to the bonsai package manager and installing the `Bonsai.ML.PointProcessDecoder` package. Additional installation steps are required for installing the CPU or GPU version of the `Bonsai.ML.Torch` package. See the [Torch installation guide](../Torch/torch-overview.md) for more information.

## Bonsai Implementation

The following workflow showcases the core functionality of the `Bonsai.ML.PointProcessDecoder` package.

:::workflow
![Point Process Decoder Implementation](~/workflows/PointProcessDecoder.bonsai)
:::

The `CreatePointProcessModel` node is used to define a model and configure it's parameters. For details on model configuration, please see the [PointProcessDecoder documentation](https://github.com/ncguilbeault/PointProcessDecoder). Crucially, the user must specify the `Name` property in the `Model Parameters` section, as this is what allows you to reference the specific model in the `Encode` and `Decode` nodes, the two main methods that the model will use.

During encoding, the user passes in a tuple of `Covariates` and `SpikeCounts`. `Covariates` are variables that the user measures (for example, the animal's position), represented as a (M, N) tensor, where M is the number of samples in a batch and N is the number of dimensions of the data. Spike counts are the data used for decoding, and can be either sorted spike data or clusterless. If the data are sorted spikes, then the `SpikeCounts` tensor will be an (M, U) tensor, where U is the number of sorted units. If the data are clusterless, then the tensor will be an (M, F, C) tensor, where F is the number of features computed for each mark (for instance, the maximum spike amplitude across electrodes), and C is the number of independant recording channels (for example, individual tetrodes). This approach is quite general, and the features used to mark spiking events can be anythink. The encoder learns the probability of observing spikes given data.

Decoding is the process of taking just the `SpikeCounts` and inferring what is the probability of observing the `Covariates`. To do this, the model uses a bayesian state-space model to predict a posterior distribution over the latent space by combining the likelihood of the spikes with a prior distribution. Depending on the decoder used, the output will either be a (M x D*) tensor, with D* representing a multivariate probability mass function over the state-space (for the `StateSpaceDecoder` type), or will be an (M x S x D*) tensor with S being the number of discrete states describing the dynamics of data (for the `HybridStateSpaceClassifier` type).