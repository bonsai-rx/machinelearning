using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the parameters of a Kalman filter model.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="KalmanFilterParameters"/> struct with the specified parameters.
/// </remarks>
/// <param name="transitionMatrix"></param>
/// <param name="measurementFunction"></param>
/// <param name="processNoiseCovariance"></param>
/// <param name="measurementNoiseCovariance"></param>
/// <param name="initialState"></param>
/// <param name="initialCovariance"></param>
public struct KalmanFilterParameters(
    Tensor transitionMatrix,
    Tensor measurementFunction,
    Tensor processNoiseCovariance,
    Tensor measurementNoiseCovariance,
    Tensor initialState,
    Tensor initialCovariance)
{
    /// <summary>
    /// The state transition matrix.
    /// </summary>
    public Tensor TransitionMatrix = transitionMatrix;

    /// <summary>
    /// The measurement function.
    /// </summary>
    public Tensor MeasurementFunction = measurementFunction;

    /// <summary>
    /// The process noise covariance.
    /// </summary>
    public Tensor ProcessNoiseCovariance = processNoiseCovariance;

    /// <summary>
    /// The measurement noise covariance.
    /// </summary>
    public Tensor MeasurementNoiseCovariance = measurementNoiseCovariance;

    /// <summary>
    /// The initial state.
    /// </summary>
    public Tensor InitialState = initialState;

    /// <summary>
    /// The initial covariance.
    /// </summary>
    public Tensor InitialCovariance = initialCovariance;
}