# Overview

The `Bonsai.ML.PointProcessDecoder` package provides a Bonsai interface to the [PointProcessDecoder package](https://github.com/ncguilbeault/PointProcessDecoder), which is a C# implementation of a Bayesian state-space point process decoder inspired by the [replay_trajectory_classification](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) algorithm. It can decode behaviour from spiking data. 

## Installation

The package can be installed by going to the bonsai package manager and installing the `Bonsai.ML.PointProcessDecoder` package. Additional installation steps are required for installing the CPU or GPU version of the `Bonsai.ML.Torch` package. See the [Torch installation guide](../Torch/torch-overview.md) for more information.

## Bonsai Implementation

The following workflow showcases the core functionality of the `Bonsai.ML.PointProcessDecoder` package.

:::workflow
![Point Process Decoder Implementation](~/workflows/PointProcessDecoder.bonsai)
:::

The `CreatePointProcessModel` node is used to instantiate a model object and configure parameters while the `Encode` and `Decode` nodes are the two main methods that will be used.

Below detailed breakdown of the configuration parameters and how they relate to model specification. 

### Model Specification

#### Model Parameters

The model parameters are used to define generic properties of the model. The `Device` property determines the torch device to run the model on. In most cases, this will be either CPU (the default configuration if no device is specified) or CUDA (for running on the GPU). The `ScalarType` property is used to set the data type of tensors computed and maintined within the model. The `Name` property is used to ascribe a unique key to a model, which allows other nodes throughout the workflow to reference the same model. In this way, multiple models can be defined and used in a single workflow so long as each model is given a unique name.  

#### Covariate Parameters

The covariate parameters relate to the covariate or observation data. The `MinRange` and `MaxRange` values determine the boundaries of our evaluation range, while `Steps` determine how many steps or bins will be used to define the range between the min and max values. The decoded posterior will be `Tensor` object whose shape depends on the `Steps` parameters, and the indices of the tensor will correspond to bins within the range. The `Dimensions` parameter determines the number of dimensions of the covariate data. It is important to note that `MinRange`, `MaxRange`, and `Steps` are arrays whose number of elements must match the number of `Dimensions` of the covariate data. 

#### Encoder Parameters

The encoder parameters help define the encoding procedure, which relates the covariate data to spiking activity in the model. The `Bandwidth` parameter is used by the model during probability density estimation of the covariate data, $\pi(x)$. A larger bandwidth means the estimation of the covariate distribution will have greater spread, while a smaller bandwidth means the distribution will be more peaked. The `EncoderType` is selected depending on whether the spiking data are organized into sorted units or whether the data are marked spike events. When `SortedSpikes` is used, the `NUnits` property is set to the number of sorted units in the spiking data. When `ClusterlessMarks` is selected, the `MarkDimensions`, `MarkChannels`, and `MarkBandwidth` parameters must be set based on the number of mark features, number of recording channels, and the kernel bandwidth of the marks, respectively. Larger values for the `MarkBandwidth` parameter correspond to greater spread in the estimated probability density of the marked spiking data.

The `EstimationMethod` determines the probability density estimation procedure used by the encoder. The `KernelDensity` method maintains a unique kernel for every data point added, which has the greatest accuracy but requires the most amount of memory. The `KernelCompression` method reduces the total number kernels in memory by computing the Mahalanobis distance between new data and existing kernels, and using the `DistanceThreshold` parameter to determine if new data will get incorporated into the closest existing kernel or whether a new kernel gets created. The lower the `DistanceThreshold`, the more likely it is that new data points will create new kernels. 

During the encoding process, new data samples are continuously added to memory for improving probability density estimation. Very quickly, the memory needed for computing density estimates can exceed the memory of the computer, which will cause an out of memory error. The `KernelLimit` parameter is an optional parameter which can be set to limit the maximum number of kernels maintained for a single estimated distribution, ensuring that the model's memory remains within the limits of the computer. When the `KernelLimit` is reached while using the `KernelDensity` method, new data points will be added in place of the oldest data points. When `KernelCompression` is used and the `KernelLimit` is reached, new data points will be merged with their closest existing kernel.

#### Decoder Parameters

The decoder parameters define the Bayesian inference procedure of the model. This procedure involves using the likelihood of the spiking data, $p(O_t|x_t)$, and the prior, $p(x_{t-1}|O_{1:t-1})$, to compute the posterior distribution, $p(x_t|O_{1:t})$. The `DecoderType` determines the model dynamics. The `StateSpaceDecoder` type incorporates continuous state transitions into the model, $p(x_t|x_{t-1})$, and the `TransitionsType` parameter defines how the latent variable evolves in time. If the `HybridStateSpaceClassifier` type is selected, the model incorporates both continuous state transitions, $p(x_t|x_{t-1},I_t,I_{t-1})$ and discrete state transitions, $p(I_t|I_{t-1})$, where the discrete states correspond to different continuous state dynamics of stationary, random walk, or uniform. The `StayProbability` parameter is used to define the probability of transitioning to the same discrete state from the previous state.

Decoding is the process of taking just the `SpikeCounts` and inferring what is the probability of observing the `Covariates`. To do this, the model uses a bayesian state-space model to predict a posterior distribution over the latent space by combining the likelihood of the spikes with a prior distribution. Depending on the decoder used, the output will either be a (M x D*) tensor, with D* representing a multivariate probability mass function over the state-space (for the `StateSpaceDecoder` type), or will be an (M x S x D*) tensor with S being the number of discrete states describing the dynamics of data (for the `HybridStateSpaceClassifier` type).

### Encoding

The encoder learns the probability of observing spikes given data. During encoding, the user passes in a tuple of `Covariates` and `SpikeCounts`. `Covariates` are variables that the user measures (for example, the animal's position), represented as a (M, N) tensor, where M is the number of samples in a batch and N is the number of dimensions of the data. If the data are sorted spikes, then the `SpikeCounts` tensor will be an (M, U) tensor, where U is the number of sorted units. If the data are clusterless, then the tensor will be an (M, F, C) tensor, where F is the number of features computed for each mark (for instance, the maximum spike amplitude across electrodes), and C is the number of independant recording channels (for example, individual tetrodes). This approach is quite general in the sense that the features used to mark spiking events can be anything that informs the model of the unique identity of spike source, and recording channels can be tetrodes, silicon probes, Neuropixels, etc.

### Decoding

Decoding is the process of inferring the latent state variable from the observed spikes alone. For this, the model uses a Bayesian state-space approach to infer a posterior distribution over the latent variable by combining the likelihood of the spikes with a prior distribution. For the `StateSpaceDecoder` type, the posterior output will be a (M x D*) tensor, with D* representing a multivariate probability mass function over the covariate evaluation range. For the `HybridStateSpaceClassifier` type, the posterior will be an (M x S x D*) tensor, with S being the 3 discrete states (stationary, random walk, and uniform) describing the continuous state dynamics of the latent variable.

## More information

For more information and documentation about the model, please see the [PointProcessDecoder repo](https://github.com/ncguilbeault/PointProcessDecoder).