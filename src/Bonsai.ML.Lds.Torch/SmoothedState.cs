using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a Kalman smoother.
/// </summary>
/// <param name="smoothedMean"></param>
/// <param name="smoothedCovariance"></param>
public struct SmoothedState(
    Tensor smoothedMean,
    Tensor smoothedCovariance) : ILdsState
{
    /// <summary>
    /// The smoothed state after the smoothing step.
    /// </summary>
    public Tensor SmoothedMean = smoothedMean;

    /// <summary>
    /// The smoothed covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedCovariance = smoothedCovariance;

    /// <inheritdoc/>
    public readonly Tensor Mean => SmoothedMean;

    /// <inheritdoc/>
    public readonly Tensor Covariance => SmoothedCovariance;
}