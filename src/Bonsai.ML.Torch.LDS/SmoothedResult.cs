using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of a Kalman smoother.
/// </summary>
/// <param name="smoothedMean"></param>
/// <param name="smoothedCovariance"></param>
/// <param name="smoothedInitialMean"></param>
/// <param name="smoothedInitialCovariance"></param>
public struct SmoothedResult(
    Tensor smoothedMean,
    Tensor smoothedCovariance,
    Tensor smoothedInitialMean = null,
    Tensor smoothedInitialCovariance = null)
{
    /// <summary>
    /// The smoothed state after the smoothing step.
    /// </summary>
    public Tensor SmoothedMean = smoothedMean;

    /// <summary>
    /// The smoothed covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedCovariance = smoothedCovariance;

    /// <summary>
    /// The smoothed initial state after the smoothing step.
    /// </summary>
    public Tensor SmoothedInitialMean = smoothedInitialMean;

    /// <summary>
    /// The smoothed initial covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedInitialCovariance = smoothedInitialCovariance;
}