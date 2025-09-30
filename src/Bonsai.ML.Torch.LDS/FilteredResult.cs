using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of a Kalman filter.
/// </summary>
/// <param name="predictedMean"></param>
/// <param name="predictedCovariance"></param>
/// <param name="updatedMean"></param>
/// <param name="updatedCovariance"></param>
public struct FilteredResult(
    Tensor predictedMean,
    Tensor predictedCovariance,
    Tensor updatedMean,
    Tensor updatedCovariance)
{
    /// <summary>
    /// The predicted mean after the prediction step.
    /// </summary>
    public Tensor PredictedMean = predictedMean;

    /// <summary>
    /// The predicted covariance after the prediction step.
    /// </summary>
    public Tensor PredictedCovariance = predictedCovariance;

    /// <summary>
    /// The updated mean after the update step.
    /// </summary>
    public Tensor UpdatedMean = updatedMean;

    /// <summary>
    /// The updated covariance after the update step.
    /// </summary>
    public Tensor UpdatedCovariance = updatedCovariance;
}