using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a linear gaussian dynamical system.
/// </summary>
/// <param name="mean"></param>
/// <param name="covariance"></param>
public readonly struct LinearDynamicalSystemState(Tensor mean, Tensor covariance) : ILinearDynamicalSystemState
{
    /// <inheritdoc/>
    public readonly Tensor Mean => mean;

    /// <inheritdoc/>
    public readonly Tensor Covariance => covariance;
}
