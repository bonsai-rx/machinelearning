using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of a Kalman smoother.
/// </summary>
/// <param name="smoothedMean"></param>
/// <param name="smoothedCovariance"></param>
public struct SmoothedResult(
    Tensor smoothedMean,
    Tensor smoothedCovariance)
{
    /// <summary>
    /// The smoothed state after the smoothing step.
    /// </summary>
    public Tensor SmoothedMean = smoothedMean;

    /// <summary>
    /// The smoothed covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedCovariance = smoothedCovariance;
}