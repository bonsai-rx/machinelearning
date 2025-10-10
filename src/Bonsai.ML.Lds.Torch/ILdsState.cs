using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a linear gaussian dynamical system.
/// </summary>
public interface ILdsState
{
    /// <summary>
    /// The mean of the state.
    /// </summary>
    Tensor Mean { get; }

    /// <summary>
    /// The covariance of the state.
    /// </summary>
    Tensor Covariance { get; }
}