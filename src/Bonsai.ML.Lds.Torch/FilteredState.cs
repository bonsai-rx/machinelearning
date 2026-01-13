using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the state of a Kalman filter.
/// </summary>
/// <param name="predictedState"></param>
/// <param name="updatedState"></param>
/// <param name="innovation"></param>
/// <param name="innovationCovariance"></param>
/// <param name="kalmanGain"></param>
/// <param name="logLikelihood"></param>
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
