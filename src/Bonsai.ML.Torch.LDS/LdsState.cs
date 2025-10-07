using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the state of a linear gaussian dynamical system.
/// </summary>
/// <param name="mean"></param>
/// <param name="covariance"></param>
public class LdsState(Tensor mean, Tensor covariance) : ILdsState
{
    /// <inheritdoc/>
    public Tensor Mean => mean;
    
    /// <inheritdoc/>
    public Tensor Covariance => covariance;
}