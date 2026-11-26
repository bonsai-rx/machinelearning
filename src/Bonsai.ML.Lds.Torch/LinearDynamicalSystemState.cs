using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a linear gaussian dynamical system.
/// </summary>
/// <param name="mean">The mean of the state.</param>
/// <param name="covariance">The covariance of the state.</param>
public readonly struct LinearDynamicalSystemState(Tensor mean, Tensor covariance) : ILinearDynamicalSystemState
{
    /// <inheritdoc/>
    public readonly Tensor Mean => mean;

    /// <inheritdoc/>
    public readonly Tensor Covariance => covariance;
}
