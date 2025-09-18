using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of a Kalman filter.
/// </summary>
/// <param name="predictedState"></param>
/// <param name="predictedCovariance"></param>
/// <param name="updatedState"></param>
/// <param name="updatedCovariance"></param>
public struct FilteredResult(
    Tensor predictedState,
    Tensor predictedCovariance,
    Tensor updatedState,
    Tensor updatedCovariance)
{
    /// <summary>
    /// The predicted state after the prediction step.
    /// </summary>
    public Tensor PredictedState = predictedState;

    /// <summary>
    /// The predicted covariance after the prediction step.
    /// </summary>
    public Tensor PredictedCovariance = predictedCovariance;

    /// <summary>
    /// The updated state after the update step.
    /// </summary>
    public Tensor UpdatedState = updatedState;

    /// <summary>
    /// The updated covariance after the update step.
    /// </summary>
    public Tensor UpdatedCovariance = updatedCovariance;
}