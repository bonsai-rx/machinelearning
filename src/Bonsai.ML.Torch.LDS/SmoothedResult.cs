using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of a Kalman smoother step.
/// </summary>
/// <param name="smoothedState"></param>
/// <param name="smoothedCovariance"></param>
/// <param name="smoothedLagOneCovariance"></param>
/// <param name="smoothedInitialState"></param>
/// <param name="smoothedInitialCovariance"></param>
public struct SmoothedResult(
    Tensor smoothedState,
    Tensor smoothedCovariance,
    Tensor smoothedLagOneCovariance,
    Tensor smoothedInitialState = null,
    Tensor smoothedInitialCovariance = null)
{
    /// <summary>
    /// The smoothed state after the smoothing step.
    /// </summary>
    public Tensor SmoothedState = smoothedState;

    /// <summary>
    /// The smoothed covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedCovariance = smoothedCovariance;

    /// <summary>
    /// The smoothed lag-one covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedLagOneCovariance = smoothedLagOneCovariance;

    /// <summary>
    /// The smoothed initial state after the smoothing step.
    /// </summary>
    public Tensor SmoothedInitialState = smoothedInitialState;

    /// <summary>
    /// The smoothed initial covariance after the smoothing step.
    /// </summary>
    public Tensor SmoothedInitialCovariance = smoothedInitialCovariance;
}