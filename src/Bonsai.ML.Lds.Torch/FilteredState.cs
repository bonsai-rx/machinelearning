using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a Kalman filter.
/// </summary>
/// <param name="predictedState">The state following the prediction step.</param>
/// <param name="updatedState">The state following the update step.</param>
/// <param name="innovation">The difference between the observation and its prediction.</param>
/// <param name="innovationCovariance">The covariance of the innovation.</param>
/// <param name="kalmanGain">The Kalman gain used in the update step.</param>
/// <param name="logLikelihood">The log likelihood of the observation.</param>
public readonly struct FilteredState(
    LinearDynamicalSystemState predictedState,
    LinearDynamicalSystemState updatedState,
    Tensor innovation = null,
    Tensor innovationCovariance = null,
    Tensor kalmanGain = null,
    Tensor logLikelihood = null) : ILinearDynamicalSystemState
{
    /// <summary>
    /// The predicted state following the prediction step.
    /// </summary>
    public readonly LinearDynamicalSystemState PredictedState => predictedState;

    /// <summary>
    /// The updated state following the update step.
    /// </summary>
    public readonly LinearDynamicalSystemState UpdatedState => updatedState;

    /// <summary>
    /// The innovation (residual) between the observation and the prediction.
    /// </summary>
    public readonly Tensor Innovation => innovation;

    /// <summary>
    /// The innovation (residual) covariance.
    /// </summary>
    public readonly Tensor InnovationCovariance => innovationCovariance;

    /// <summary>
    /// The Kalman gain.
    /// </summary>
    public readonly Tensor KalmanGain => kalmanGain;

    /// <summary>
    /// The log likelihood of the observation given the updated state.
    /// </summary>
    public readonly Tensor LogLikelihood => logLikelihood;

    /// <inheritdoc/>
    public readonly Tensor Mean => updatedState.Mean;

    /// <inheritdoc/>
    public readonly Tensor Covariance => updatedState.Covariance;
}
