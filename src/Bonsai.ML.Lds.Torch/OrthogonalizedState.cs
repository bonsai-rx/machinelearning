using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of an LDS after orthogonalizing the state mean and covariance estimates.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="OrthogonalizedState"/> struct.
/// </remarks>
/// <param name="orthogonalizedMean"></param>
/// <param name="orthogonalizedCovariance"></param>
public struct OrthogonalizedState(
    Tensor orthogonalizedMean,
    Tensor orthogonalizedCovariance) : ILdsState
{
    /// <summary>
    /// The orthogonalized mean estimate.
    /// </summary>
    public Tensor OrthogonalizedMean = orthogonalizedMean;

    /// <summary>
    /// The orthogonalized covariance estimate.
    /// </summary>
    public Tensor OrthogonalizedCovariance = orthogonalizedCovariance;

    /// <inheritdoc/>
    public readonly Tensor Mean => OrthogonalizedMean;

    /// <inheritdoc/>
    public readonly Tensor Covariance => OrthogonalizedCovariance;
}