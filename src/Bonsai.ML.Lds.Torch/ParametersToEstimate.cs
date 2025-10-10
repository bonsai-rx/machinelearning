namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the parameters to estimate for a Kalman filter model.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="KalmanFilterParameters"/> struct with the specified parameters.
/// </remarks>
/// <param name="transitionMatrix"></param>
/// <param name="measurementFunction"></param>
/// <param name="processNoiseCovariance"></param>
/// <param name="measurementNoiseCovariance"></param>
/// <param name="initialMean"></param>
/// <param name="initialCovariance"></param>
public struct ParametersToEstimate(
    bool transitionMatrix = true,
    bool measurementFunction = true,
    bool processNoiseCovariance = true,
    bool measurementNoiseCovariance = true,
    bool initialMean = true,
    bool initialCovariance = true)
{
    /// <summary>
    /// The state transition matrix.
    /// </summary>
    public bool TransitionMatrix = transitionMatrix;

    /// <summary>
    /// The measurement function.
    /// </summary>
    public bool MeasurementFunction = measurementFunction;

    /// <summary>
    /// The process noise covariance.
    /// </summary>
    public bool ProcessNoiseCovariance = processNoiseCovariance;

    /// <summary>
    /// The measurement noise covariance.
    /// </summary>
    public bool MeasurementNoiseCovariance = measurementNoiseCovariance;

    /// <summary>
    /// The initial mean.
    /// </summary>
    public bool InitialMean = initialMean;

    /// <summary>
    /// The initial covariance.
    /// </summary>
    public bool InitialCovariance = initialCovariance;
}