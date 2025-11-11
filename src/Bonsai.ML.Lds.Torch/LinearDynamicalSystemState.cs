using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a linear gaussian dynamical system.
/// </summary>
/// <param name="mean"></param>
/// <param name="covariance"></param>
public class LinearDynamicalSystemState(Tensor mean, Tensor covariance) : ILinearDynamicalSystemState
{
    /// <inheritdoc/>
    public Tensor Mean => mean;
    
    /// <inheritdoc/>
    public Tensor Covariance => covariance;
}